# Running InsTaG on RunPod

This document provides instructions for running the InsTaG framework on RunPod, a cloud platform offering GPU instances.

## Overview

InsTaG (Learning Personalized 3D Talking Head from Few-Second Video) is a framework that creates realistic 3D talking head avatars from very short videos. The RunPod setup provides:

- A ready-to-use Docker image with all dependencies pre-installed
- Support for both interactive (via Terminal) and API-based usage
- Combined environments for both the main InsTaG framework and the Sapiens geometry prior generation

## Getting Started

### Option 1: Using the Template

1. Go to RunPod.io and select the InsTaG template from the template gallery
2. Choose your desired GPU type (recommend at least 16GB VRAM)
3. Start the pod
4. Connect via SSH or HTTPS Terminal

### Option 2: Custom Deployment

1. Go to RunPod.io and deploy a GPU pod
2. Select the "Docker" deployment option
3. Specify the Docker image: `your-registry/instag-runpod:latest`
4. Start the pod
5. Connect via SSH or HTTPS Terminal

## Using InsTaG on RunPod

### Interactive Mode

Once connected to your pod, you can use InsTaG commands directly:

1. **Process a video**:
   ```bash
   # First, upload your video to the pod
   # Example: Place it at /app/data/john/john.mp4
   
   python data_utils/process.py /app/data/john/john.mp4
   ```

2. **Generate teeth masks**:
   ```bash
   export PYTHONPATH=./data_utils/easyportrait
   python data_utils/easyportrait/create_teeth_mask.py /app/data/john
   ```

3. **Generate geometry priors** (optional, for very short videos):
   ```bash
   # Switch to the Sapiens environment
   conda activate sapiens_lite
   
   # Run Sapiens
   bash data_utils/sapiens/run.sh /app/data/john
   
   # Switch back to main environment
   conda activate instag
   ```

4. **Fine-tune the model**:
   ```bash
   bash scripts/train_xx_few.sh /app/data/john /app/output/john_model 0
   ```

5. **Generate synthesis**:
   ```bash
   python synthesize_fuse.py -S /app/data/john -M /app/output/john_model --audio /path/to/audio.wav --audio_extractor esperanto
   ```

### API Mode

The container includes a RunPod handler that exposes InsTaG functionality via the RunPod API:

```python
# Python example of calling the RunPod API
import requests

API_URL = "https://api.runpod.ai/v2/YOUR_POD_ID/run"
API_KEY = "YOUR_API_KEY"

def process_video(video_path):
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "input": {
                "operation": "process_video",
                "video_path": video_path
            }
        }
    )
    return response.json()

def generate_teeth_mask(person_dir):
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "input": {
                "operation": "generate_teeth_mask",
                "person_dir": person_dir
            }
        }
    )
    return response.json()

def run_sapiens(person_dir):
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "input": {
                "operation": "run_sapiens",
                "person_dir": person_dir
            }
        }
    )
    return response.json()

def fine_tune(data_dir, output_dir, gpu_id="0"):
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "input": {
                "operation": "fine_tune",
                "data_dir": data_dir,
                "output_dir": output_dir,
                "gpu_id": gpu_id
            }
        }
    )
    return response.json()

def synthesize(args):
    response = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "input": {
                "operation": "synthesize",
                "args": args
            }
        }
    )
    return response.json()
```

## Data Management

### Uploading Data

You can upload data to your RunPod instance using:

1. **RunPod Volume**: Attach a volume to your pod during creation and place data there
2. **SFTP**: Use SFTP to upload files to your pod
3. **Cloud Storage**: Download data from S3, Google Drive, etc. using commands like `wget` or `curl`

### Downloading Results

1. **RunPod Volume**: Output is saved on the volume if attached
2. **SFTP**: Download files via SFTP
3. **Cloud Storage**: Upload results to S3, Google Drive, etc.

## Working with the Basel Face Model

The InsTaG framework requires the Basel Face Model 2009 (BFM) for face tracking. Due to licensing, it's not included in the Docker image:

1. Register at [Basel Face Model website](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads)
2. Download the 01_MorphableModel.mat file
3. Upload it to your pod at: `/app/data_utils/face_tracking/3DMM/01_MorphableModel.mat`
4. Convert the model:
   ```bash
   cd /app/data_utils/face_tracking
   python convert_BFM.py
   ```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce batch size by editing training scripts
   - Use a GPU with more VRAM
   - Clear PyTorch cache: `rm -rf ~/.cache/torch`

2. **CUDA Errors**:
   - Ensure you're using a compatible NVIDIA GPU
   - Verify CUDA works: `python -c "import torch; print(torch.cuda.is_available())"`

3. **Missing Models**:
   - If model downloading fails, you may need to manually download them
   - See the prepare.sh script for download URLs

For additional help, refer to the InsTaG GitHub repository and documentation.

## Reference

- [Official InsTaG Repository](https://github.com/Fictionarry/InsTaG)
- [RunPod Documentation](https://docs.runpod.io/)