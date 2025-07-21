# InsTaG Docker Setup

This document provides instructions for running InsTaG using Docker and Docker Compose for containerized training and inference.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your system
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed (for GPU support)
- An NVIDIA GPU with sufficient VRAM (12+ GB recommended)
- NVIDIA drivers compatible with CUDA 11.7
- The [Basel Face Model (BFM2009)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads) (requires registration)

## Quick Start

1. Clone the repository (if you haven't already):
   ```bash
   git clone https://github.com/Fictionarry/InsTaG.git
   cd InsTaG
   git submodule update --init --recursive
   ```

2. Run the setup script to build containers and download required resources:
   ```bash
   chmod +x setup-docker.sh
   ./setup-docker.sh
   ```

3. Download the Basel Face Model:
   - Register at [Basel Face Model website](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads)
   - Download the 01_MorphableModel.mat file
   - Place it at `data_utils/face_tracking/3DMM/01_MorphableModel.mat`
   - Convert the model:
     ```bash
     ./docker-run.sh convert-bfm
     ```

## Container Architecture

The Docker setup consists of two separate containers:

1. **Main InsTaG Container (`instag`):**
   - Based on CUDA 11.7 with Python 3.9
   - Contains PyTorch 1.13.1, TensorFlow 2.10.0, OpenFace
   - Used for all training, processing, and inference tasks
   
2. **Sapiens Container (`sapiens`):**
   - Based on CUDA 12.1 with Python 3.10
   - Contains PyTorch 2.2.1
   - Used specifically for generating geometry priors for short videos
   - Only needed if you want to use Sapiens for improved fine-tuning on very short videos

This dual-container approach is necessary because Sapiens requires a different Python and PyTorch version than the main InsTaG framework.

## Complete Training Workflow

### Pre-Training (Identity-Free Stage)

1. Place pre-training videos in the data directory:
   ```bash
   mkdir -p data/pretrain/person1
   cp /path/to/video.mp4 data/pretrain/person1/person1.mp4
   ```

2. Process each video to extract frames and audio:
   ```bash
   ./docker-run.sh process data/pretrain/person1/person1.mp4
   ```

3. Generate teeth masks:
   ```bash
   ./docker-run.sh teeth-mask data/pretrain/person1
   ```

4. Extract facial Action Units:
   ```bash
   ./docker-run.sh extract-au data/pretrain/person1
   ```

5. Run pre-training:
   ```bash
   ./docker-run.sh pretrain data/pretrain output/pretrain_model 0
   ```
   This will train the universal motion field on all videos in data/pretrain.

### Adaptation (Person-Specific Stage)

1. Place a video of the target person:
   ```bash
   mkdir -p data/alice
   cp /path/to/alice_video.mp4 data/alice/alice.mp4
   ```

2. Process the video:
   ```bash
   ./docker-run.sh process data/alice/alice.mp4
   ```

3. Generate teeth masks:
   ```bash
   ./docker-run.sh teeth-mask data/alice
   ```

4. Extract facial Action Units:
   ```bash
   ./docker-run.sh extract-au data/alice
   ```

5. For short videos (< 10 seconds), generate geometry priors:
   ```bash
   ./docker-run.sh run-sapiens data/alice
   ```

6. Fine-tune the model:
   ```bash
   ./docker-run.sh train data/alice output/alice_model 0
   ```

7. Synthesize with new audio:
   ```bash
   ./docker-run.sh synthesize -S data/alice -M output/alice_model --audio path_to_audio.wav --audio_extractor deepspeech
   ```

## Audio Feature Options

InsTaG supports multiple audio feature extractors, each with different characteristics:

1. **DeepSpeech** (default):
   - Basic speech features
   - Example:
     ```bash
     ./docker-run.sh extract-audio-features data/alice/audio.wav deepspeech
     ./docker-run.sh synthesize -S data/alice -M output/alice_model --audio_extractor deepspeech
     ```

2. **Wav2Vec**:
   - Better lip synchronization
   - Example:
     ```bash
     ./docker-run.sh extract-audio-features data/alice/audio.wav wav2vec
     ./docker-run.sh synthesize -S data/alice -M output/alice_model --audio_extractor esperanto
     ```

3. **AVE** (Audio-Visual Entangler):
   - Best lip-sync quality for English
   - Example:
     ```bash
     # AVE features are processed on-the-fly
     ./docker-run.sh synthesize -S data/alice -M output/alice_model --audio audio.wav --audio_extractor ave
     ```

4. **HuBERT**:
   - Good for non-English languages
   - Example:
     ```bash
     ./docker-run.sh extract-audio-features data/alice/audio.wav hubert
     ./docker-run.sh synthesize -S data/alice -M output/alice_model --audio_extractor hubert
     ```

## Available Commands

Run `./docker-run.sh` without arguments to see the complete list of available commands:

```
Usage: ./docker-run.sh COMMAND [ARGS]

Available commands:
  build                      - Build the Docker image
  build-sapiens              - Build the Sapiens Docker image
  shell                      - Open a shell in the container
  sapiens-shell              - Open a shell in the Sapiens container
  prepare                    - Run the prepare.sh script inside the container
  prepare-sapiens            - Run the prepare_sapiens.sh script
  download-easyportrait-model - Download the EasyPortrait model
  convert-bfm                - Convert Basel Face Model (requires manual download first)
  pretrain ARGS              - Run pretrain_con.sh with arguments (data source, output dir, gpu)
  train ARGS                 - Run train_df_few.sh with arguments (data source, output dir, gpu)
  process VIDEO_PATH         - Process a video using data_utils/process.py
  teeth-mask PATH            - Generate teeth masks for a given person directory
  extract-au PATH            - Extract facial Action Units for a person using OpenFace
  extract-audio-features PATH TYPE - Extract audio features (types: deepspeech, wav2vec, hubert, ave)
  run-sapiens PATH           - Generate Sapiens geometry priors for a given person
  synthesize ARGS            - Run synthesize_fuse.py with arguments
```

## Different Training Scenarios

### Training on Very Short Videos (5-10 seconds)

For very short videos, Sapiens geometry priors are essential:

```bash
# Process the short video
./docker-run.sh process data/john/john.mp4

# Generate teeth masks and extract AUs
./docker-run.sh teeth-mask data/john
./docker-run.sh extract-au data/john

# Generate geometry priors with Sapiens
./docker-run.sh run-sapiens data/john

# Fine-tune with default settings
./docker-run.sh train data/john output/john_model 0
```

### Training on Longer Videos (>30 seconds)

For longer videos, you can skip geometry priors and use the "--long" flag:

```bash
# Process the video
./docker-run.sh process data/mary/mary.mp4

# Generate teeth masks and extract AUs
./docker-run.sh teeth-mask data/mary
./docker-run.sh extract-au data/mary

# Open a shell to edit the training script
./docker-run.sh shell

# Inside the container:
# Edit scripts/train_xx_few.sh to add --long flag to the python commands
# Then exit and run:
./docker-run.sh train data/mary output/mary_model 0
```

## Customization

### Modifying the Dockerfile

If you need to customize the Docker environment:

1. Edit the `Dockerfile` (for main container) or `Dockerfile.sapiens` (for Sapiens container) with your changes
2. Rebuild the image with `./docker-run.sh build` or `./docker-run.sh build-sapiens` respectively

### Using a Different CUDA Version

The default configuration uses CUDA 11.7 for the main container and CUDA 12.1 for the Sapiens container. To use a different CUDA version:

1. Edit the `Dockerfile` to change the base image (e.g., to `nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04`)
2. Update the environment file reference to use the appropriate file (e.g., `environment.yml` for CUDA 11.3)
3. Rebuild the image

## Troubleshooting

### Common Issues

- **Docker build failures with conda activation**:
  - If you encounter errors like `process "/bin/sh -c conda create -n instag python=3.9 -y && echo \"source activate instag\" > ~/.bashrc && . /opt/conda/etc/profile.d/conda.sh && conda activate instag && conda install pytorch..."` did not complete successfully, this is due to conda activation issues in non-interactive shells.
  - Fix: Use `conda run -n instag` instead of activating the environment with `conda activate instag` in the Dockerfile.

- **"Unable to find teeth mask" error**:
  - Make sure you've downloaded the EasyPortrait model:
    ```bash
    ./docker-run.sh download-easyportrait-model
    ```
  - Verify the model exists at `data_utils/easyportrait/fpn-fp-512.pth`

- **OpenFace FeatureExtraction failures**:
  - Make sure your video frames have clear faces visible
  - Try with fewer frames initially (use a shorter video)
  - Run in the shell for detailed output:
    ```bash
    ./docker-run.sh shell
    # Inside container:
    FeatureExtraction -fdir data/person/frames -out_dir data/person/au -aus
    ```

- **PyTorch3D installation failures**:
  - PyTorch3D may fail to install depending on the PyTorch version
  - The container will still work for most use cases without PyTorch3D
  - If needed, install it manually in the container following their installation guide

- **GPU not visible in container**:
  - Ensure the NVIDIA Container Toolkit is properly installed
  - Verify your drivers are compatible with CUDA 11.7
  - Test with `nvidia-smi` on the host
  - Inside the container, run:
    ```bash
    ./docker-run.sh shell
    # Inside container:
    python -c "import torch; print(torch.cuda.is_available())"
    ```

- **Out of memory errors during training**:
  - Reduce batch size in training scripts
  - Use a smaller value for `--init_num` in training scripts
  - Free up space by removing cached files:
    ```bash
    ./docker-run.sh shell
    # Inside container:
    rm -rf ~/.cache/torch
    ```

### Handling Submodule Compilation Errors

If you encounter issues with the CUDA submodules:

1. Enter the container shell:
   ```bash
   ./docker-run.sh shell
   ```

2. Manually install the problematic module:
   ```bash
   cd /app/submodules/diff-gaussian-rasterization
   pip uninstall -y diff_gaussian_rasterization
   pip install -e .
   
   cd /app/submodules/simple-knn
   pip uninstall -y simple-knn
   pip install -e .
   
   cd /app/gridencoder
   pip uninstall -y gridencoder
   pip install -e .
   ```

## Notes

- The containers mount `./data`, `./output`, and `./scripts` directories from your host machine, ensuring that your data and results persist outside the container
- All model weights and training results will be saved to the `./output` directory
- To download the Basel Face Model (BFM2009), you'll need to register on their website and follow the instructions in the training document
- For multi-GPU training, use `CUDA_VISIBLE_DEVICES` in the training scripts or specify a different GPU index in the training commands 

## CI Builds vs Full Installation

The Dockerfile includes special handling for GitHub Actions CI builds:

- OpenFace installation is skipped in the CI environment to speed up builds
- PyTorch3D installation is optional and allowed to fail
- The prepare.sh script can be skipped if necessary

When building locally or for production, you may want to set the `CI=false` environment variable to ensure all components are installed:

```bash
CI=false docker build -t instag:latest .
```

For the full experience including OpenFace, you'll need to run the container and manually install OpenFace:

```bash
docker run --gpus all -it instag:latest /bin/bash
# Then inside the container:
git clone https://github.com/TadasBaltrusaitis/OpenFace.git /tmp/OpenFace
cd /tmp/OpenFace
bash ./download_models.sh
mkdir -p build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE ..
make -j4
make install
cp -r /tmp/OpenFace/build/bin /instag/OpenFace/
cp -r /tmp/OpenFace/lib /instag/OpenFace/ 
cp -r /tmp/OpenFace/build/lib /instag/OpenFace/
```