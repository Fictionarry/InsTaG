#!/bin/bash
set -e

# Text colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}==============================================${NC}"
echo -e "${BLUE}   InsTaG Docker Container Builder v1.0       ${NC}"
echo -e "${BLUE}==============================================${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if NVIDIA Docker is installed
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}NVIDIA drivers not detected. GPU support may not work properly.${NC}"
    read -p "Continue anyway? (y/n): " choice
    if [[ "$choice" != "y" ]]; then
        exit 1
    fi
fi

# Ensure we're in the right directory
cd "$(dirname "$0")"

echo -e "${GREEN}Step 1: Building InsTaG Docker image (this may take 30-60 minutes)${NC}"
echo "Building image with tag instag:latest..."
echo ""

# Start the build
docker build -t instag:latest -t instag:1.3.0 . || {
    echo -e "${YELLOW}Build failed. Check the errors above.${NC}"
    exit 1
}

echo ""
echo -e "${GREEN}Step 2: Setting up data directory${NC}"
echo ""

# Check if data directory exists
if [ ! -d "./data" ]; then
    echo "Creating data directory..."
    mkdir -p ./data
else
    echo "Data directory already exists."
fi

# Check if output directory exists
if [ ! -d "./output" ]; then
    echo "Creating output directory..."
    mkdir -p ./output
else
    echo "Output directory already exists."
fi

echo ""
echo -e "${GREEN}Step 3: Testing container${NC}"
echo ""

# Test run the container
echo "Running a test to verify the container works..."
docker run --rm --gpus all -it instag:latest python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch version:', torch.__version__)" || {
    echo -e "${YELLOW}Container test failed. CUDA might not be properly configured.${NC}"
    read -p "Continue anyway? (y/n): " choice
    if [[ "$choice" != "y" ]]; then
        exit 1
    fi
}

echo ""
echo -e "${GREEN}Build completed successfully!${NC}"
echo ""
echo -e "${BLUE}Usage instructions:${NC}"
echo "1. Place your videos in the './data/' directory"
echo "2. For processing videos, run: ./docker-run.sh process data/<ID>/<ID>.mp4"
echo "3. For training, run: ./docker-run.sh train data/<ID> output/<project_name> 0"
echo ""
echo -e "${BLUE}For more information, see README_docker.md${NC}"
echo ""