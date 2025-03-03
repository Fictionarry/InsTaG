#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import torch
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import pandas as pd
import glob

from utils.sh_utils import SH2RGB
from utils.audio_utils import get_audio_features
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    background: np.array
    talking_dict: dict

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".jpg", audio_file='', audio_extractor='deepspeech', N_views=-1, preload=True):
    cam_infos = []
    postfix_dict = {"deepspeech": "", "esperanto": "_eo", "hubert": "_hu"}
    
    N_views = N_views if "train" in transformsfile and audio_file=='' else -1

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        focal_len = contents["focal_len"]

        frames = contents["frames"][:N_views]            
        
        if audio_extractor == "ave":
            from torch.utils.data import DataLoader
            from scene.motion_net import AudioEncoder
            from utils.audio_utils import AudDataset
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = AudioEncoder().to(device).eval()
            ckpt = torch.load('./data_utils/audio_visual_encoder.pth')
            model.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
            read_cache = False
            if audio_file == '':
                if os.path.exists(os.path.join(path, 'aud_ave.npy')):
                    aud_features = np.load(os.path.join(path, 'aud_ave.npy'))
                    print(aud_features.shape)
                    read_cache = True
                else:
                    dataset = AudDataset(os.path.join(path, 'aud.wav')) 
            else:
                dataset = AudDataset(audio_file)
            if not read_cache:
                data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
                outputs = []
                for mel in tqdm(data_loader):
                    mel = mel.to(device)
                    with torch.no_grad():
                        out = model(mel)
                    outputs.append(out)
                outputs = torch.cat(outputs, dim=0).cpu()
                first_frame, last_frame = outputs[:1], outputs[-1:]
                aud_features = torch.cat([first_frame.repeat(2, 1), outputs, last_frame.repeat(2, 1)],
                                            dim=0).unsqueeze(0).permute(1, 2, 0).numpy()
                if audio_file == '':
                    np.save(os.path.join(path, 'aud_ave.npy'), aud_features)
            # aud_features = np.load(os.path.join(self.root_path, 'aud_ave.npy'))
        elif audio_file == '':
            aud_features = np.load(os.path.join(path, 'aud{}.npy'.format(postfix_dict[audio_extractor])))
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features)
        aud_features = aud_features.float().permute(0, 2, 1)
        auds = aud_features
        
        # loop
        if audio_file != '':
            loop_time = auds.shape[0] // len(frames) + 1
            frames *= loop_time
        

        au_info=pd.read_csv(os.path.join(path, 'au.csv'))
        au_blink = au_info[' AU45_r'].values
        au25 = au_info[' AU25_r'].values
        au25 = np.clip(au25[:N_views], 0, np.percentile(au25[:N_views], 95))

        au25_25, au25_50, au25_75, au25_100 = np.percentile(au25, 25), np.percentile(au25, 50), np.percentile(au25, 75), au25.max()

        au_exp = []
        for i in [1,4,5,6,7,45]:
            _key = ' AU' + str(i).zfill(2) + '_r'
            au_exp_t = au_info[_key].values
            if i == 45:
                au_exp_t = au_exp_t.clip(0, 2)
            au_exp.append(au_exp_t[:, None])
        au_exp = np.concatenate(au_exp, axis=-1, dtype=np.float32)

        ldmks_lips = []
        ldmks_mouth = []
        ldmks_lhalf = []
        
        for idx, frame in tqdm(enumerate(frames)):
            lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms')) # [68, 2]
            lips = slice(48, 60)
            mouth = slice(60, 68)
            xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
            ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

            ldmks_lips.append([int(xmin), int(xmax), int(ymin), int(ymax)])
            ldmks_mouth.append([int(lms[mouth, 1].min()), int(lms[mouth, 1].max())])

            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            # self.face_rect.append([xmin, xmax, ymin, ymax])
            ldmks_lhalf.append([lh_xmin, lh_xmax, ymin, ymax])
            
        ldmks_lips = np.array(ldmks_lips)
        ldmks_mouth = np.array(ldmks_mouth)
        ldmks_lhalf = np.array(ldmks_lhalf)
        mouth_lb = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).min()
        mouth_ub = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).max()



        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, 'gt_imgs', str(frame["img_id"]) + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            talking_dict = {}
            talking_dict['img_id'] = frame['img_id']
            
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            talking_dict['image_path'] = image_path
            if preload or idx==0:
                image = Image.open(image_path)
                w, h = image.size[0], image.size[1]
                image = np.array(image.convert("RGB"))

            torso_img_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
            bg_img_path = os.path.join(path, 'bc.jpg')
            talking_dict['torso_img_path'] = torso_img_path
            talking_dict['bg_img_path'] = bg_img_path
            if preload:
                torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
                bg_img = np.array(Image.open(bg_img_path).convert("RGB"))
                bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (1 - torso_img[..., 3:] / 255.0)
                bg = bg.astype(np.uint8)
                
            if not preload:
                image = bg = None

            teeth_mask_path = os.path.join(path, 'teeth_mask', str(frame['img_id']) + '.npy')
            mask_path = os.path.join(path, 'parsing', str(frame['img_id']) + '.png')
            talking_dict['teeth_mask_path'] = teeth_mask_path
            talking_dict['mask_path'] = mask_path
            if preload:
                teeth_mask = np.load(teeth_mask_path)
                mask = np.array(Image.open(mask_path).convert("RGB")) * 1.0
                talking_dict['face_mask'] = (mask[:, :, 2] > 254) * (mask[:, :, 0] == 0) * (mask[:, :, 1] == 0) ^ teeth_mask
                talking_dict['hair_mask'] = (mask[:, :, 0] < 1) * (mask[:, :, 1] < 1) * (mask[:, :, 2] < 1)
                talking_dict['mouth_mask'] = (mask[:, :, 0] == 100) * (mask[:, :, 1] == 100) * (mask[:, :, 2] == 100) + teeth_mask

            
            if audio_file == '':
                talking_dict['auds'] = get_audio_features(auds, 2, frame['img_id'])
                if frame['img_id'] > auds.shape[0]:
                    print("[warnining] audio feature is too short")
                    break
            else:
                talking_dict['auds'] = get_audio_features(auds, 2, idx)
                if idx >= auds.shape[0]:
                    break


            talking_dict['blink'] = torch.as_tensor(np.clip(au_blink[frame['img_id']], 0, 2) / 2)
            talking_dict['au25'] = [au25[frame['img_id']], au25_25, au25_50, au25_75, au25_100]

            talking_dict['au_exp'] = torch.as_tensor(au_exp[frame['img_id']])


            [xmin, xmax, ymin, ymax] = ldmks_lips[idx].tolist()
            # padding to H == W
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            l = max(xmax - xmin, ymax - ymin) // 2
            xmin = cx - l
            xmax = cx + l
            ymin = cy - l
            ymax = cy + l

            talking_dict['lips_rect'] = [xmin, xmax, ymin, ymax]
            talking_dict['lhalf_rect'] = ldmks_lhalf[idx]
            talking_dict['mouth_bound'] = [mouth_lb, mouth_ub, ldmks_mouth[idx, 1] - ldmks_mouth[idx, 0]]
            talking_dict['img_id'] = frame['img_id']
            
            if "train" in transformsfile and N_views > 0:
                normal_path_candidates = glob.glob(os.path.join(path, 'sapiens/normal/sapiens_*'))
                normal_path_candidates.sort(reverse = True)
                normal_path = os.path.join(normal_path_candidates[0], str(frame['img_id']) + '.npy')
                talking_dict['normal_path'] = normal_path
                
                depth_path_candidates = glob.glob(os.path.join(path, 'sapiens/depth/sapiens_*'))
                depth_path_candidates.sort(reverse = True)
                depth_path = os.path.join(depth_path_candidates[0], str(frame['img_id']) + '.npy')
                talking_dict['depth_path'] = depth_path
                if preload:
                    normal = np.load(normal_path)
                    talking_dict['normal'] = torch.as_tensor(normal).permute(2, 0, 1)
                    depth = np.load(depth_path)
                    talking_dict['depth'] = torch.as_tensor(depth)
            
            FovX = focal2fov(focal_len, w)
            FovY = focal2fov(focal_len, h)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h, background=bg, talking_dict=talking_dict))
            
            # if idx > 6500: break
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".jpg", args=None, preload=True, all_for_train=False):
    audio_file = args.audio
    audio_extractor = args.audio_extractor
    if not eval:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, audio_file, audio_extractor, args.N_views, preload=preload)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_val.json", white_background, extension, audio_file, audio_extractor, args.N_views, preload=preload)
    
    if all_for_train:
        train_cam_infos.extend(test_cam_infos)
        # test_cam_infos = []
    
    
    if eval: train_cam_infos = test_cam_infos # for getNerfppNorm

    nerf_normalization = getNerfppNorm(train_cam_infos) 


    ply_path = os.path.join(path, "points3d.ply")
    mesh_path = os.path.join(path, "track_params.pt")      
    if not os.path.exists(ply_path) or True:
        # Since this data set has no colmap data, we start with random points
        num_pts = args.init_num
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 0.2 - 0.1
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        
        # facial_mesh = torch.load(mesh_path)["vertices"]
        # average_facial_mesh = torch.mean(facial_mesh, dim=0)
        # xyz = average_facial_mesh.cpu().numpy() * 0.1
        # shs = np.random.random((xyz.shape[0], 3)) / 255.0
        # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))        

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": None,
    "Blender" : readNerfSyntheticInfo
}
