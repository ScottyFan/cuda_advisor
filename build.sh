#!/bin/bash

# Build script for GridAdvisor

set -e  # Exit on error

echo "================================"
echo "Building CUDA GridAdvisor"
echo "================================"
echo ""

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo "================================"
echo "✓ Build completed successfully!"
echo "================================"
echo ""
echo "Run the test with:"
echo "  ./build/test_device_query"
echo ""