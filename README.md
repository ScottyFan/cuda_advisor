# Add to current session
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify
nvcc --version

# Clean and build
rm -rf build/
./build.sh

./build/collect_data > data/profiling_results.csv

python3 python/analyze_results.py
python3 python/compare_experiments.py

# Test each experiments
./run_experiments.sh

# Analyze all experiments
./analyze_all.sh


# CUDA GridAdvisor

> Intelligent grid/block configuration optimizer for CUDA kernels
