# Version: 1.2.0 (Build Fix)
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
FROM $BASE_IMAGE

VOLUME [ "/instag" ]

# Install system dependencies
RUN apt-get update -yq --fix-missing \
 && DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
    git \
    wget \
    cmake \
    build-essential \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libopencv-dev \
    libgtk-3-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsndfile1 \
    portaudio19-dev \
    ninja-build \
    git-lfs \
    vim \
    curl \
    libopenexr-dev \
    openexr \
    python3-dev \
    libffi-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
 && rm Miniconda3-latest-Linux-x86_64.sh

# Add conda to PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Initialize conda in bash
RUN conda init bash

# Clone InsTaG repository
RUN git lfs install \
 && git clone https://github.com/Fictionarry/InsTaG.git /instag \
 && cd /instag \
 && git submodule update --init --recursive

# Set up conda environment for InsTaG
WORKDIR /instag
RUN conda config --append channels conda-forge \
 && conda config --append channels nvidia \
 && conda create -n instag python=3.9 cudatoolkit=11.7 pytorch=1.13.1 torchvision=0.14.1 torchaudio -c pytorch -c nvidia -y \
 && echo "source activate instag" > ~/.bashrc

# Print debug information
RUN conda run -n instag python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# Install dependencies for InsTaG
RUN conda run -n instag pip install -r requirements.txt

# Install MMCV with specific CUDA version
RUN conda run -n instag pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

# Install CUDA submodules
RUN conda run -n instag bash -c "cd /instag/submodules/diff-gaussian-rasterization && FORCE_CUDA=1 pip install -e ."
RUN conda run -n instag bash -c "cd /instag/submodules/simple-knn && FORCE_CUDA=1 pip install -e ."
RUN conda run -n instag bash -c "cd /instag/gridencoder && pip install -e ."
RUN conda run -n instag bash -c "cd /instag/shencoder && pip install -e ."

# Install PyTorch3D dependencies
RUN conda run -n instag pip install "fvcore>=0.1.5" "iopath>=0.1.7" "nvidiacub-dev"

# Try to install PyTorch3D from source, but don't fail if it doesn't work
RUN conda run -n instag pip install "pytorch3d==0.7.4" || echo "PyTorch3D installation failed, but continuing build"

# Install TensorFlow
RUN conda run -n instag pip install tensorflow-gpu==2.10.0

# Skip OpenFace installation in CI environments for speed (can be installed manually later)
RUN mkdir -p /instag/OpenFace/bin

# Create a dummy OpenFace executable so scripts don't fail 
RUN echo '#!/bin/bash\necho "OpenFace not installed in this container. Please install manually if needed."' > /instag/OpenFace/bin/FeatureExtraction \
 && chmod +x /instag/OpenFace/bin/FeatureExtraction

# Download EasyPortrait model
RUN mkdir -p /instag/data_utils/easyportrait \
 && conda run -n instag wget -O /instag/data_utils/easyportrait/fpn-fp-512.pth \
    https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/easyportrait/experiments/models/fpn-fp-512.pth

# Run prepare script to download required models (continue even if it fails)
RUN cd /instag && bash scripts/prepare.sh || echo "Prepare script failed, but continuing build"

# Create the Sapiens lite environment
RUN conda create -n sapiens_lite python=3.10 -y \
 && conda run -n sapiens_lite conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y \
 && conda run -n sapiens_lite pip install opencv-python tqdm json-tricks

# Create directories for data and outputs
RUN mkdir -p /instag/data /instag/output /instag/jobs

# Set up environment paths
ENV PATH="/opt/conda/bin:/instag/OpenFace/bin:${PATH}"

# Create startup script to activate environment
RUN echo '#!/bin/bash' > /instag/startup.sh \
 && echo 'echo "Welcome to InsTaG on RunPod!"' >> /instag/startup.sh \
 && echo 'echo ""' >> /instag/startup.sh \
 && echo 'echo "Available environment commands:"' >> /instag/startup.sh \
 && echo 'echo "conda activate instag    - Activate the main InsTaG environment"' >> /instag/startup.sh \
 && echo 'echo "conda activate sapiens_lite - Activate the Sapiens environment for geometry priors"' >> /instag/startup.sh \
 && echo 'echo ""' >> /instag/startup.sh \
 && echo 'echo "Common workflows:"' >> /instag/startup.sh \
 && echo 'echo "1. Process a video:        python data_utils/process.py data/<ID>/<ID>.mp4"' >> /instag/startup.sh \
 && echo 'echo "2. Generate teeth masks:   python data_utils/easyportrait/create_teeth_mask.py ./data/<ID>"' >> /instag/startup.sh \
 && echo 'echo "3. Run Sapiens (optional): bash data_utils/sapiens/run.sh ./data/<ID>"' >> /instag/startup.sh \
 && echo 'echo "4. Fine-tune the model:    bash scripts/train_xx_few.sh data/<ID> output/<project_name> <GPU_ID>"' >> /instag/startup.sh \
 && echo 'echo "5. Synthesize:            python synthesize_fuse.py -S data/<ID> -M output/<project_name> --audio <path> --audio_extractor <type>"' >> /instag/startup.sh \
 && echo 'echo ""' >> /instag/startup.sh \
 && echo 'source /opt/conda/etc/profile.d/conda.sh' >> /instag/startup.sh \
 && echo 'conda activate instag' >> /instag/startup.sh \
 && echo 'exec bash' >> /instag/startup.sh \
 && chmod +x /instag/startup.sh

# Set working directory
WORKDIR /instag

# Default command
CMD ["/instag/startup.sh"]