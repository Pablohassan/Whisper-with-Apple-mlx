#!/bin/bash

# Script to clean up model weights before pushing to GitHub

echo "Cleaning up model weights..."

# Remove local MLX models directory (6.4GB)
if [ -d "./mlx_models" ]; then
    echo "Removing ./mlx_models directory"
    rm -rf ./mlx_models
fi

# Remove any cache directories
if [ -d "./.cache" ]; then
    echo "Removing ./.cache directory"
    rm -rf ./.cache
fi

# Find and remove any stray model files
find . -type f -name "*.bin" -o -name "*.pt" -o -name "*.pth" -o -name "*.onnx" -o -name "*.mlmodel" -o -name "*.ckpt" -o -name "*.safetensors" | while read file; do
    echo "Removing model file: $file"
    rm -f "$file"
done

# Find and remove any model directories
find . -type d -name "*model*" -not -path "*/\.*" -not -path "*/node_modules/*" | while read dir; do
    echo "Removing model directory: $dir"
    rm -rf "$dir"
done

# Create empty directories that might be needed
mkdir -p public/uploads

echo "Cleanup complete. You can now safely push to GitHub." 