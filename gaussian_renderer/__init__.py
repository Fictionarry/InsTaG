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

import torch
import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.motion_net import MotionNetwork, MouthMotionNetwork
from utils.sh_utils import eval_sh


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3Ds_precomp = cov3D_precomp,
        extra_attrs = torch.ones_like(opacity)
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "depth": rendered_depth, 
            "alpha": rendered_alpha,
            "normal": rendered_norm,
            "radii": radii}


def render_motion(viewpoint_camera, pc : GaussianModel, motion_net : MotionNetwork, pipe, bg_color : torch.Tensor, \
                    scaling_modifier=1.0, frame_idx=None, return_attn=False, personalized=False, align=False, detach_motion=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    audio_feat = viewpoint_camera.talking_dict["auds"].cuda()
    exp_feat = viewpoint_camera.talking_dict["au_exp"].cuda()
    
    xyz = pc.get_xyz

    if personalized or align:
        p_motion_preds = pc.neural_motion_grid(pc.get_xyz, audio_feat, exp_feat)
    
    if align:
        xyz = xyz + p_motion_preds['p_xyz']
        # pass
    
    motion_preds = motion_net(xyz, audio_feat, exp_feat)
    
    
    d_xyz = motion_preds['d_xyz']
    d_scale = motion_preds['d_scale']
    d_rot = motion_preds['d_rot']
        
    if personalized:
        # d_xyz *= (1 + p_motion_preds['p_scale'])
        d_xyz += p_motion_preds['d_xyz']
        d_scale += p_motion_preds['d_scale']
        d_rot += p_motion_preds['d_rot']
    
    if align:
        d_xyz *= p_motion_preds['p_scale']
        
    if detach_motion:
        d_xyz = d_xyz.detach()
        d_scale = d_scale.detach()
        d_rot = d_rot.detach()

    means3D = pc.get_xyz + d_xyz
    means2D = screenspace_points
    # opacity = pc.opacity_activation(pc._opacity + motion_preds['d_opa'])
    opacity = pc.get_opacity

    cov3D_precomp = None
    # scales = pc.get_scaling
    scales = pc.scaling_activation(pc._scaling + d_scale)
    rotations = pc.rotation_activation(pc._rotation + d_rot)

    colors_precomp = None
    shs = pc.get_features


    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3Ds_precomp = cov3D_precomp,
        extra_attrs = torch.ones_like(opacity)
    )
    
    # Attn
    rendered_attn = p_rendered_attn = None
    if return_attn:
        attn_precomp = torch.cat([motion_preds['ambient_aud'], motion_preds['ambient_eye'], torch.zeros_like(motion_preds['ambient_eye'])], dim=-1)
        rendered_attn, _, _, _, _, _ = rasterizer(
            means3D = means3D.detach(),
            means2D = means2D,
            shs = None,
            colors_precomp = attn_precomp,
            opacities = opacity.detach(),
            scales = scales.detach(),
            rotations = rotations.detach(),
            cov3Ds_precomp = cov3D_precomp,
            extra_attrs = torch.ones_like(opacity)
        )
        
        if personalized:
            p_attn_precomp = torch.cat([p_motion_preds['ambient_aud'], p_motion_preds['ambient_eye'], torch.zeros_like(p_motion_preds['ambient_eye'])], dim=-1)
            p_rendered_attn, _, _, _, _, _ = rasterizer(
                means3D = means3D.detach(),
                means2D = means2D,
                shs = None,
                colors_precomp = p_attn_precomp,
                opacities = opacity.detach(),
                scales = scales.detach(),
                rotations = rotations.detach(),
                cov3Ds_precomp = cov3D_precomp,
                extra_attrs = torch.ones_like(opacity)
            )


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "depth": rendered_depth, 
            "alpha": rendered_alpha,
            "normal": rendered_norm,
            "radii": radii,
            "motion": motion_preds,
            "p_motion": p_motion_preds if personalized or align else None,
            'attn': rendered_attn,
            "p_attn": p_rendered_attn}


def render_motion_mouth_con(viewpoint_camera, pc : GaussianModel, motion_net : MouthMotionNetwork, pc_face : GaussianModel, motion_net_face : MotionNetwork, pipe, bg_color : torch.Tensor, \
                        scaling_modifier=1.0, frame_idx=None, return_attn=False, personalized=False, align=False, k=10, inference=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    audio_feat = viewpoint_camera.talking_dict["auds"].cuda()
   
    xyz = pc.get_xyz
    
    if personalized or align:
        p_motion_preds = pc.neural_motion_grid(pc.get_xyz, audio_feat)

    if align:
        xyz = xyz + p_motion_preds['p_xyz']
        # pass
    
    if not inference:
        exp_feat = viewpoint_camera.talking_dict["au_exp"].cuda()
        exp_feat = torch.zeros_like(exp_feat)
        motion_preds_face = motion_net_face(pc_face.get_xyz, audio_feat, exp_feat)
    else:
        motion_preds_face = motion_net_face.cache
        
    with torch.no_grad():
        motion_max, _ = motion_preds_face["d_xyz"][..., 1].topk(k, 0, True, True)
        motion_min, _ =  motion_preds_face["d_xyz"][..., 1].topk(k, 0, False, True)
        move_feat = torch.as_tensor([[motion_max[-1], motion_min[-1], motion_max[-1] - motion_min[-1]]]).cuda() * 1e2
        
    motion_preds = motion_net(xyz, audio_feat, move_feat.detach())
    d_xyz = motion_preds['d_xyz']
    # d_rot = motion_preds['d_rot']
    
    if personalized:
        # d_xyz *= (1 + p_motion_preds['p_scale'])
        d_xyz += p_motion_preds['d_xyz']
        # d_rot += p_motion_preds['d_rot']
                
    means3D = pc.get_xyz + d_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    cov3D_precomp = None
    scales = pc.get_scaling
    rotations = pc.rotation_activation(pc._rotation) # + d_rot)

    colors_precomp = None
    shs = pc.get_features

    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3Ds_precomp = cov3D_precomp,
        extra_attrs = torch.ones_like(opacity)
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "depth": rendered_depth, 
            "alpha": rendered_alpha,
            "radii": radii,
            "motion": motion_preds,
            "p_motion": p_motion_preds if personalized or align else None
            }

