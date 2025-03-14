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
    && rm -rf /var/lib/apt/lists/*

# Set up interactive shell
SHELL ["/bin/bash", "-i", "-c"]

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && sh Miniconda3-latest-Linux-x86_64.sh -b -u -p ~/miniconda3 \
 && ~/miniconda3/bin/conda init \
 && source ~/.bashrc \
 && rm Miniconda3-latest-Linux-x86_64.sh

# Set up environment for InsTaG
RUN conda create -n instag python=3.9 -y \
 && conda activate instag \
 && conda install pytorch==1.13.1 torchvision==0.14.1 cudatoolkit=11.7 -c pytorch -y

# Clone InsTaG repository
RUN git lfs install \
 && git clone https://github.com/Fictionarry/InsTaG.git /instag \
 && cd /instag \
 && git submodule update --init --recursive

# Install dependencies for InsTaG
WORKDIR /instag
RUN conda activate instag \
 && pip install -r requirements.txt \
 && cd /instag/submodules/diff-gaussian-rasterization && FORCE_CUDA=1 pip install -e . \
 && cd /instag/submodules/simple-knn && FORCE_CUDA=1 pip install -e . \
 && cd /instag/gridencoder && pip install -e . \
 && cd /instag/shencoder && pip install -e . \
 && pip install "git+https://github.com/facebookresearch/pytorch3d.git" \
 && pip install tensorflow-gpu==2.10.0

# Install OpenFace
RUN conda activate instag \
 && git clone https://github.com/TadasBaltrusaitis/OpenFace.git /tmp/OpenFace \
 && cd /tmp/OpenFace \
 && bash ./download_models.sh \
 && mkdir -p build \
 && cd build \
 && cmake -D CMAKE_BUILD_TYPE=RELEASE .. \
 && make -j4 \
 && make install \
 && cp -r /tmp/OpenFace/build/bin /instag/OpenFace \
 && cp -r /tmp/OpenFace/lib /instag/OpenFace/ \
 && cp -r /tmp/OpenFace/build/lib /instag/OpenFace/ \
 && rm -rf /tmp/OpenFace

# Download EasyPortrait model
RUN conda activate instag \
 && mkdir -p /instag/data_utils/easyportrait \
 && wget -O /instag/data_utils/easyportrait/fpn-fp-512.pth \
    https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/easyportrait/experiments/models/fpn-fp-512.pth

# Run prepare script to download required models
RUN conda activate instag \
 && cd /instag \
 && bash scripts/prepare.sh

# Create the Sapiens lite environment
RUN conda create -n sapiens_lite python=3.10 -y \
 && conda activate sapiens_lite \
 && conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.7 -c pytorch -c nvidia \
 && pip install opencv-python tqdm json-tricks

# Create directories for data and outputs
RUN mkdir -p /instag/data /instag/output /instag/jobs

# Set up environment paths
ENV PATH="/root/miniconda3/bin:/instag/OpenFace/bin:${PATH}"

# Set up startup script
RUN echo 'echo "Welcome to InsTaG on RunPod\!"' > /instag/startup.sh \
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
 && echo 'exec bash' >> /instag/startup.sh \
 && chmod +x /instag/startup.sh

# Set working directory
WORKDIR /instag

# Default command
CMD ["/instag/startup.sh"]
