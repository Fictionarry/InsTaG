#!/bin/bash
# Script to set up the Docker environment for InsTaG

# Ensure script exits on error
set -e

# Print header
echo "=================================="
echo "InsTaG Docker Environment Setup"
echo "=================================="
echo ""

# Check Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH."
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose is installed 
if ! docker compose version &> /dev/null; then
    echo "Error: Docker Compose is not installed or not in PATH."
    echo "Please install Docker Compose first: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check NVIDIA Docker is installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: NVIDIA drivers may not be installed."
    echo "GPU support requires NVIDIA drivers and NVIDIA Container Toolkit."
    echo "For more information: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Make docker-run.sh executable
chmod +x docker-run.sh

# Create necessary directories if they don't exist
echo "Creating data and output directories..."
mkdir -p data/pretrain
mkdir -p output
mkdir -p data_utils/face_tracking/3DMM
mkdir -p data_utils/easyportrait
mkdir -p submodules

# Check for submodules
if [ ! -d "submodules/diff-gaussian-rasterization" ] || [ ! -d "submodules/simple-knn" ]; then
    echo "Initializing git submodules..."
    git submodule update --init --recursive
fi

# Build the main container
echo "Building main InsTaG container (this may take a while)..."
./docker-run.sh build

# Ask if user wants to build Sapiens container
read -p "Do you want to build the Sapiens container for geometry priors? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Building Sapiens container (this may take a while)..."
    ./docker-run.sh build-sapiens
fi

# Ask if user wants to download required models
read -p "Do you want to download required models and resources? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading models and resources..."
    ./docker-run.sh prepare

    # Download EasyPortrait model
    read -p "Do you want to download the EasyPortrait model for teeth masking? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Downloading EasyPortrait model..."
        ./docker-run.sh download-easyportrait-model
    fi
    
    # Ask about Sapiens models
    read -p "Do you want to download Sapiens models (required for geometry priors)? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Downloading Sapiens models..."
        ./docker-run.sh prepare-sapiens
    fi
fi

# Prompt about Basel Face Model
echo ""
echo "NOTE: The Basel Face Model (BFM2009) is required for face tracking."
echo "You need to manually download it from https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads"
echo "After downloading, place the file at: data_utils/face_tracking/3DMM/01_MorphableModel.mat"
echo "Then run: ./docker-run.sh convert-bfm"
echo ""

# Print completion message
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "You can now use the docker-run.sh script to interact with the InsTaG environment."
echo "For a list of available commands, run:"
echo "./docker-run.sh"
echo ""
echo "Next steps:"
echo "1. Download the Basel Face Model (BFM2009) if you haven't already"
echo "2. Place a pretrain video in data/pretrain/<person>/<person>.mp4"
echo "3. Process the video with: ./docker-run.sh process data/pretrain/<person>/<person>.mp4"
echo "4. Generate teeth masks: ./docker-run.sh teeth-mask data/pretrain/<person>"
echo "5. Extract facial Action Units: ./docker-run.sh extract-au data/pretrain/<person>"
echo ""
echo "For more information, refer to README_docker.md" 