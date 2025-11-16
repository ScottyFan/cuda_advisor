# Add to current session
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify
nvcc --version

# Clean and build
rm -rf build/
./build.sh

[wf2060@cuda5 GPU]$ export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
[wf2060@cuda5 GPU]$ ls
benchmarks  build  build.sh  CMakeLists.txt  data  docs  include  python  README.md  src  tests
[wf2060@cuda5 GPU]$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.2, V12.2.140
Build cuda_12.2.r12.2/compiler.33191640_0


# CUDA GridAdvisor

> Intelligent grid/block configuration optimizer for CUDA kernels

## What We Built (So Far)

A tool that automatically recommends optimal CUDA kernel launch configurations.

### ✅ Completed Components

1. **Device Query** - Extracts GPU specifications
2. **Profiling Harness** - Measures kernel performance accurately
3. **Benchmark Suite** - 10+ diverse kernels (memory/compute/balanced)
4. **Feature Extraction** - Extracts 24-dimensional feature vectors
5. **Analytical Model** - Predicts optimal configurations using occupancy/roofline

### 🎯 Current Capabilities

- Extract hardware specs from any NVIDIA GPU
- Profile kernels with different thread configurations
- Extract 24 features from compiled kernels
- Predict optimal thread counts using analytical model
- Measure speedup vs baseline configurations

## Quick Start

### Build
```bash
./build.sh
```

### Run Tests
```bash
./build/test_device_query    # Test GPU detection
./build/test_profiler        # Test profiling system
./build/test_benchmarks      # Test all benchmarks
./build/test_features        # Test feature extraction
./build/gridadvisor          # Test complete system
```

### Example Output
```
GridAdvisor Recommendation:
  Threads/Block: 256
  Num Blocks:    16384
  Occupancy:     85%
  Reasoning:     Memory-bound: Balancing occupancy and coalescing
```

## Project Structure
```
cuda-gridadvisor/
├── include/
│   ├── utils/
│   │   ├── device_query.hpp      # GPU specs
│   │   ├── profiler.hpp          # Performance measurement
│   │   └── profiler_impl.hpp
│   └── features/
│       ├── feature_extractor.hpp  # Feature extraction
│       └── feature_extractor_impl.hpp
├── src/
│   ├── utils/
│   │   └── device_query.cu
│   ├── test_device_query.cu
│   ├── test_profiler.cu
│   ├── test_benchmarks.cu
│   ├── test_features.cu
│   └── gridadvisor.cu            # Main tool
├── benchmarks/
│   ├── memory_bound.cu           # Transpose, reduction, scatter
│   ├── compute_bound.cu          # MatMul, Mandelbrot
│   └── balanced.cu               # Stencil, histogram, scan
├── CMakeLists.txt
└── build.sh
```

## Features Extracted (24 total)

### Hardware (5)
- SM count, threads/SM, memory bandwidth, peak GFlops, shared memory

### Kernel Resources (6)
- Registers/thread, shared memory, local memory, instruction count

### Compute Characteristics (5)
- Arithmetic intensity, FLOP count, memory ops, FP ratio

### Memory Access (4)
- Access pattern, shared memory reuse, load ratio

### Control Flow (2)
- Branch intensity, sync intensity

### Problem Size (2)
- Total work items, work per SM

## How It Works

1. **Extract Features** - Analyze kernel using CUDA Driver API
2. **Calculate Occupancy** - For each candidate configuration
3. **Score Configurations** - Based on kernel type (compute/memory bound)
4. **Select Best** - Highest scoring configuration
5. **Return Recommendation** - With reasoning

## Results So Far

Tested on RTX 4070:
- ✅ Device query working
- ✅ Profiling accurate (sub-millisecond precision)
- ✅ 10+ benchmarks running correctly
- ✅ Feature extraction successful
- ✅ Analytical predictions reasonable

## Tomorrow's Tasks

### Phase 1: Data Collection (Optional)
- [ ] Run profiler on all benchmarks
- [ ] Sweep all configurations
- [ ] Save results to CSV
- [ ] Collect ~2000+ data points

### Phase 2: ML Training (Optional)
- [ ] Python script for XGBoost training
- [ ] Feature scaling
- [ ] Model validation
- [ ] Save trained model

### Phase 3: Testing & Evaluation
- [ ] Test on all benchmarks
- [ ] Compare analytical vs actual performance
- [ ] Measure accuracy metrics
- [ ] Generate plots

### Phase 4: Report
- [ ] Write methodology
- [ ] Document results
- [ ] Create presentation slides
- [ ] Prepare demo

## File Count: 19/20 ✅

Keeping it minimal and focused!

## Requirements

- CUDA 11.0+
- CMake 3.18+
- GCC 7.5+
- NVIDIA GPU

## Team

Scott Fan

## Course

NYU GPU Architecture - Fall 2025
Prof. Mohamed Zahran
