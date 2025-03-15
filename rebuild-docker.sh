#!/bin/bash

echo "Rebuilding InsTaG Docker image from scratch..."
echo ""

# Remove any existing images
echo "Removing existing instag images..."
docker rmi instag:latest || true
docker rmi instag:cuda11.7 || true

# Clear the docker build cache
echo "Clearing Docker build cache..."
docker builder prune -f

# Rebuild the image with no cache
echo "Building fresh image (no cache)..."
docker build --no-cache -t instag:latest -t instag:cuda11.7 .

echo ""
echo "Build complete. Run './docker-run.sh shell' to test."
echo ""