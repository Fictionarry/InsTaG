FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Prevent timezone questions during package installations
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.9 \
    python3.9-dev \
    python3-pip \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsndfile1 \
    portaudio19-dev \
    build-essential \
    cmake \
    libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
    && python -m pip install --upgrade pip

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# Add conda to path
ENV PATH="/opt/conda/bin:${PATH}"

# Create a working directory
WORKDIR /app

# First, copy only the environment file to leverage Docker caching
COPY environment_cu117.yml /app/

# Create conda environment
RUN conda env create -f environment_cu117.yml

# Make the conda environment the default
SHELL ["conda", "run", "-n", "instag", "/bin/bash", "-c"]

# Install OpenFace for facial action unit extraction
RUN git clone https://github.com/TadasBaltrusaitis/OpenFace.git /tmp/OpenFace \
    && cd /tmp/OpenFace \
    && bash ./download_models.sh \
    && mkdir -p build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE .. \
    && make -j4 \
    && make install \
    && cp -r /tmp/OpenFace/build/bin /app/OpenFace \
    && cp -r /tmp/OpenFace/lib /app/OpenFace/ \
    && cp -r /tmp/OpenFace/build/lib /app/OpenFace/ \
    && rm -rf /tmp/OpenFace

# Install additional required dependencies
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git" || \
    echo "PyTorch3D installation failed, please check compatibility with PyTorch version" \
    && pip install tensorflow-gpu==2.10.0 \
    && pip install openmim \
    && mim install mmcv-full==1.7.1 prettytable

# Copy the repository (except for large data files)
COPY . /app/

# Properly initialize and install submodules in one step to avoid race conditions
RUN git submodule update --init --recursive \
    && cd /app/submodules/diff-gaussian-rasterization && pip install -e . \
    && cd /app/submodules/simple-knn && pip install -e . \
    && cd /app/gridencoder && pip install -e .

# Create directories for data and output
RUN mkdir -p /app/data /app/output

# Add a script to activate the conda environment when starting the container
RUN echo '#!/bin/bash\neval "$(conda shell.bash hook)"\nconda activate instag\nexec "$@"' > /app/entrypoint.sh \
    && chmod +x /app/entrypoint.sh

# Add OpenFace to PATH
ENV PATH="/app/OpenFace/bin:${PATH}"

ENTRYPOINT ["/app/entrypoint.sh"]

# Default command keeps the container running
CMD ["bash"] 