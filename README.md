# CUDA GridAdvisor

> Intelligent grid/block configuration optimizer for CUDA kernels using analytical modeling

## 📋 Project Overview

GridAdvisor is a tool that **predicts optimal thread/block configurations** for CUDA kernels without exhaustive search. It uses:
- **24-dimensional feature extraction** from kernel characteristics
- **Analytical classification model** based on kernel categories
- **Automated benchmarking** across different problem sizes

**Current Performance:** 25-50% prediction accuracy across different problem sizes (see Results section)

---

## 🏗️ Project Structure

```
cuda-gridadvisor/
├── benchmarks/           # CUDA benchmark kernels
│   ├── memory_bound.cu   # Transpose, Reduction, Scatter
│   ├── compute_bound.cu  # MatMul, Mandelbrot
│   └── balanced.cu       # Stencil, Histogram, Prefix Sum
├── include/
│   ├── utils/            # Device query, profiler
│   └── features/         # Feature extraction (24-dim vector)
├── src/
│   ├── collect_data.cu   # Main data collection + prediction model
│   ├── test_*.cu         # Individual component tests
│   └── utils/            # Device query implementation
├── python/               # Analysis scripts
│   ├── analyze_results.py      # Single experiment analysis
│   └── compare_experiments.py  # Cross-size comparison
├── data/                 # Experimental results & analysis
└── CMakeLists.txt        # Build configuration
```

---

## 🚀 Quick Start

### Prerequisites
```bash
# CUDA Toolkit (11.0+)
nvcc --version

# Python 3 with dependencies
pip install pandas numpy matplotlib
```

### Setup Environment
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Build
```bash
# Clean build
rm -rf build/
./build.sh
```

---

## 🧪 Running Experiments

### Option 1: Run All Experiments (Recommended)
Run across 4 problem sizes (small/medium/large/xlarge):
```bash
./run_experiments.sh
```
**Output:** `data/profiling_{small,medium,large,xlarge}.csv`

**Note:** This modifies `src/collect_data.cu` automatically for each size, then rebuilds.

---

### Option 2: Run Single Experiment

#### Step 1: Edit Problem Size
Edit `src/collect_data.cu` and modify these lines:

```cpp
// For transpose/stencil kernels
const int width = 2048, height = 2048;  // Change to: 1024, 2048, 4096, 8192

// For reduction/scatter/histogram
const int N = 1 << 24;  // Change to: 1<<20, 1<<22, 1<<24, 1<<26

// For matmul kernels  
const int M = 512, N = 512, K = 512;  // Change to: 256, 512, 1024, 2048
```

**Example sizes:**
| Size   | Transpose | Reduction | MatMul |
|--------|-----------|-----------|--------|
| Small  | 1024x1024 | 1<<20     | 256    |
| Medium | 2048x2048 | 1<<24     | 512    |
| Large  | 4096x4096 | 1<<26     | 1024   |
| XLarge | 8192x8192 | 1<<28     | 2048   |

#### Step 2: Rebuild & Run
```bash
cd build
make -j$(nproc)
cd ..

./build/collect_data > data/my_experiment.csv 2> data/my_experiment.log
```

**Check logs:**
```bash
cat data/my_experiment.log
```

---

### Option 3: Test Individual Components

#### Test Device Query
```bash
./build/test_device_query
```

#### Test Feature Extraction
```bash
./build/test_features
```

#### Test Profiler
```bash
./build/test_profiler
```

#### Test All Benchmarks
```bash
./build/test_benchmarks
```

#### Run GridAdvisor Predictor
```bash
./build/gridadvisor
```

---

## 📊 Analyzing Results

### Single Experiment Analysis
```bash
python3 python/analyze_results.py \
    -i data/profiling_medium.csv \
    -o data/analysis_medium
```

**Output:**
- `analysis_medium/performance_curves.png` - Time vs thread count
- `analysis_medium/accuracy_by_kernel.png` - Prediction accuracy bar chart
- `analysis_medium/occupancy_vs_time.png` - Occupancy correlation
- `analysis_medium/analysis_report.txt` - Text summary

### Compare All Experiments
```bash
python3 python/compare_experiments.py
```

**Output:** `data/comparison/` with scalability analysis

### Batch Analysis
```bash
./analyze_all.sh  # Analyzes all profiling_*.csv files
```

---

## 📈 Current Results

### Accuracy by Problem Size
| Problem Size | Accuracy | Correct/Total |
|--------------|----------|---------------|
| Small        | 25.0%    | 2/8           |
| Medium       | 25.0%    | 2/8           |
| **Large**    | **50.0%**| **4/8** ⭐    |
| XLarge       | 37.5%    | 3/8           |

### Per-Kernel Performance
| Kernel         | Small | Medium | Large | XLarge | Notes |
|----------------|-------|--------|-------|--------|-------|
| matmul_naive   | ✅    | ✅     | ✅    | ✅     | Most stable |
| reduction      | ❌    | ❌     | ✅    | ✅     | Size-sensitive |
| transpose      | ❌    | ❌     | ❌    | ✅     | Needs high threads |
| stencil_2d     | ✅    | ❌     | ✅    | ❌     | Unpredictable |
| histogram      | ❌    | ✅     | ✅    | ❌     | Mid-range bias |
| mandelbrot     | ✅    | ❌     | ❌    | ❌     | Compute-heavy |
| scatter        | ❌    | ❌     | ❌    | ❌     | Always wrong |
| prefix_sum     | ❌    | ❌    | ❌    | ❌     | Always wrong |

---

## 🔧 Model Details

### Feature Vector (24 dimensions)
1. **Hardware (5):** SM count, threads/SM, bandwidth, peak GFlops, shared mem
2. **Kernel Resources (6):** Registers, shared mem, local mem, const mem, instructions
3. **Compute (5):** Arithmetic intensity, FLOP count, memory ops, FP ratio
4. **Memory Access (4):** Access pattern, shared mem reuse, coalescing
5. **Control Flow (2):** Branch intensity, sync intensity
6. **Problem Size (2):** Total work items, work per SM

### Prediction Model (Model V4)
**5 Kernel Categories:**
1. **Memory Bandwidth Limited** → Prefer 512 threads (e.g., transpose)
2. **Compute Heavy** → Prefer 256-512 threads (e.g., mandelbrot)
3. **Shared Memory Bound** → Prefer 128 threads (e.g., reduction)
4. **Register Limited** → Prefer 32-64 threads (e.g., scatter, matmul_naive)
5. **Balanced** → Prefer 256 threads (e.g., stencil)

**Classification Rules:**
- Uses register count, shared memory usage, arithmetic intensity
- Calculates occupancy for each candidate (32, 64, 128, 256, 512, 1024)
- Scores based on category + occupancy + secondary factors

---

## 🐛 Known Issues

1. **Low accuracy for scatter/prefix_sum** - Access pattern detection needs work
2. **Problem size sensitivity** - Model doesn't adapt well across different sizes
3. **Feature estimation** - Some features (f10, f11) use placeholder values
4. **Occupancy vs Performance** - High occupancy doesn't always mean best performance

---

## 🔮 Future Improvements

### To Boost Accuracy to 80%+:
1. **Better feature extraction:**
   - Use CUDA profiling APIs (nvprof/nsight) for real metrics
   - Add memory coalescing efficiency measurements
   
2. **Machine Learning approach:**
   - Replace analytical model with Random Forest/XGBoost
   - Train on larger dataset with more GPU types
   
3. **Problem size awareness:**
   - Add size-dependent rules (e.g., transpose: if size > 4096x4096, prefer 512)
   - Normalize features by problem size
   
4. **Per-GPU tuning:**
   - Different GPUs may have different optimal configs
   - Add GPU architecture features (compute capability, memory hierarchy)

---

## 📝 Citation

This project is based on NYU GPU Computing Course - Project 6:
> Mohamed Zahran, "Graphics Processing Units (GPUs): Architecture and Programming"

---

## 📧 Contact

For questions or improvements, feel free to reach out or submit an issue!
