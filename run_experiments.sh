#!/bin/bash

echo "================================"
echo "GridAdvisor - Automated Experiments"
echo "================================"
echo ""

# Create data directory
mkdir -p data

# Initial build
echo "Building project initially..."
./build.sh
if [ $? -ne 0 ]; then
    echo "✗ Initial build failed!"
    exit 1
fi
echo ""

# Experiment 1: Small
echo "[1/5] Running Small Problem Size..."
echo "  Modifying source files..."
sed -i 's/const int N = [0-9]*/const int N = 1024/g' src/collect_data.cu
sed -i 's/const int N = 1 << [0-9]*/const int N = 1 << 20/g' src/collect_data.cu
sed -i 's/const int M = [0-9]*, N = [0-9]*, K = [0-9]*/const int M = 256, N = 256, K = 256/g' src/collect_data.cu
sed -i 's/const int width = [0-9]*, height = [0-9]*/const int width = 1024, height = 1024/g' src/collect_data.cu

echo "  Rebuilding..."
cd build && make -j$(nproc) && cd ..
if [ $? -ne 0 ]; then
    echo "✗ Build failed!"
    exit 1
fi

echo "  Running benchmark..."
./build/collect_data > data/profiling_small.csv 2> data/log_small.txt
echo "✓ Small completed"
echo ""

# Experiment 2: Medium
echo "[2/5] Running Medium Problem Size..."
echo "  Modifying source files..."
sed -i 's/const int N = [0-9]*/const int N = 2048/g' src/collect_data.cu
sed -i 's/const int N = 1 << [0-9]*/const int N = 1 << 24/g' src/collect_data.cu
sed -i 's/const int M = [0-9]*, N = [0-9]*, K = [0-9]*/const int M = 512, N = 512, K = 512/g' src/collect_data.cu
sed -i 's/const int width = [0-9]*, height = [0-9]*/const int width = 2048, height = 2048/g' src/collect_data.cu

echo "  Rebuilding..."
cd build && make -j$(nproc) && cd ..
if [ $? -ne 0 ]; then
    echo "✗ Build failed!"
    exit 1
fi

echo "  Running benchmark..."
./build/collect_data > data/profiling_medium.csv 2> data/log_medium.txt
echo "✓ Medium completed"
echo ""

# Experiment 3: Large
echo "[3/5] Running Large Problem Size..."
echo "  Modifying source files..."
sed -i 's/const int N = [0-9]*/const int N = 4096/g' src/collect_data.cu
sed -i 's/const int N = 1 << [0-9]*/const int N = 1 << 26/g' src/collect_data.cu
sed -i 's/const int M = [0-9]*, N = [0-9]*, K = [0-9]*/const int M = 1024, N = 1024, K = 1024/g' src/collect_data.cu
sed -i 's/const int width = [0-9]*, height = [0-9]*/const int width = 4096, height = 4096/g' src/collect_data.cu

echo "  Rebuilding..."
cd build && make -j$(nproc) && cd ..
if [ $? -ne 0 ]; then
    echo "✗ Build failed!"
    exit 1
fi

echo "  Running benchmark..."
./build/collect_data > data/profiling_large.csv 2> data/log_large.txt
echo "✓ Large completed"
echo ""

# Experiment 4: XLarge
echo "[4/5] Running XLarge Problem Size..."
echo "  Modifying source files..."
sed -i 's/const int N = [0-9]*/const int N = 8192/g' src/collect_data.cu
sed -i 's/const int N = 1 << [0-9]*/const int N = 1 << 28/g' src/collect_data.cu
sed -i 's/const int M = [0-9]*, N = [0-9]*, K = [0-9]*/const int M = 2048, N = 2048, K = 2048/g' src/collect_data.cu
sed -i 's/const int width = [0-9]*, height = [0-9]*/const int width = 8192, height = 8192/g' src/collect_data.cu

echo "  Rebuilding..."
cd build && make -j$(nproc) && cd ..
if [ $? -ne 0 ]; then
    echo "✗ Build failed!"
    exit 1
fi

echo "  Running benchmark..."
./build/collect_data > data/profiling_xlarge.csv 2> data/log_xlarge.txt
echo "✓ XLarge completed"
echo ""

# Restore to medium (default)
echo "[5/5] Restoring to medium size..."
sed -i 's/const int N = [0-9]*/const int N = 2048/g' src/collect_data.cu
sed -i 's/const int N = 1 << [0-9]*/const int N = 1 << 24/g' src/collect_data.cu
sed -i 's/const int M = [0-9]*, N = [0-9]*, K = [0-9]*/const int M = 512, N = 512, K = 512/g' src/collect_data.cu
sed -i 's/const int width = [0-9]*, height = [0-9]*/const int width = 2048, height = 2048/g' src/collect_data.cu
cd build && make -j$(nproc) && cd ..

echo ""
echo "================================"
echo "✓ All experiments completed!"
echo "================================"
echo ""
echo "Generated files:"
ls -lh data/profiling_*.csv
echo ""
echo "Check logs in data/log_*.txt for any errors"
