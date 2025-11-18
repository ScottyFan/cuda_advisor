#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#include "utils/device_query.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace gridadvisor {

// ========== NEW: Profiling Results ==========
struct ProfilingMetrics {
    // Runtime metrics from test execution
    float achieved_occupancy;
    float memory_throughput_pct;      // % of peak bandwidth
    float gld_efficiency;              // Global load efficiency (0-1)
    float gst_efficiency;              // Global store efficiency (0-1)
    int total_instructions;            // Actual instruction count
    float branch_efficiency;           // % branches not diverged
    float shared_mem_replay_overhead;  // Replay count for shared mem
    
    // Derived metrics
    float effective_bandwidth_gb_s;
    float memory_intensity;            // Bytes per instruction
};

// Enhanced feature vector with real profiling data
struct FeatureVector {
    // Hardware features (5)
    float f1_sm_count;
    float f2_max_threads_per_sm;
    float f3_memory_bandwidth_gb_s;
    float f4_peak_gflops;
    float f5_shared_memory_per_sm_kb;
    
    // Kernel resource features (6) - from cudaFuncAttributes
    float f6_registers_per_thread;
    float f7_shared_memory_per_block_kb;
    float f8_local_memory_per_thread_b;
    float f9_const_memory_kb;
    float f10_instruction_count;          // NOW REAL!
    float f11_binary_size_kb;             // NOW REAL!
    
    // Compute characteristics (5) - from profiling
    float f12_arithmetic_intensity;
    float f13_flop_count;                 // Estimated from instructions
    float f14_memory_ops;                 // From profiling
    float f15_fp_ratio;                   // FP instructions / total
    float f16_transcendental_ratio;       // sin/cos/exp ratio
    
    // Memory access features (4) - from profiling
    float f17_access_pattern_score;       // NOW REAL! (gld_efficiency)
    float f18_shared_mem_reuse;           // From replay overhead
    float f19_global_load_ratio;          // loads / (loads + stores)
    float f20_memory_divergence;          // 1 - branch_efficiency
    
    // Control flow features (2)
    float f21_branch_intensity;
    float f22_sync_intensity;
    
    // Problem size features (2)
    float f23_total_work_items;
    float f24_work_per_sm;
    
    void print() const;
    std::vector<float> to_array() const;
};

struct KernelCharacteristics {
    std::string name;
    int registers_per_thread;
    int shared_memory_static;
    int shared_memory_dynamic;
    int const_memory;
    int local_memory;
    int max_threads_per_block;
    size_t binary_size;                    // NEW!
    
    enum class Type {
        MEMORY_BOUND,
        COMPUTE_BOUND,
        BALANCED,
        UNKNOWN
    } type;
    
    bool uses_shared_memory;
    bool has_atomics;
    float estimated_ai;
};

class FeatureExtractor {
public:
    FeatureExtractor(int device_id = 0);
    
    // Main extraction with profiling
    template<typename KernelFunc, typename... Args>
    FeatureVector extract_with_profiling(
        KernelFunc kernel, 
        int problem_size,
        Args... args  // Actual kernel arguments for test run
    );
    
    // Legacy method (backward compatible)
    template<typename KernelFunc>
    FeatureVector extract(KernelFunc kernel, int problem_size);
    
    // Get kernel characteristics
    template<typename KernelFunc>
    KernelCharacteristics get_kernel_characteristics(KernelFunc kernel);
    
private:
    int device_id;
    DeviceSpecs device_specs;
    
    // NEW: Profiling helpers
    template<typename KernelFunc, typename... Args>
    ProfilingMetrics run_profiling_test(
        KernelFunc kernel,
        int test_size,
        Args... args
    );
    
    void estimate_compute_features(
        FeatureVector& fv, 
        const KernelCharacteristics& kc,
        const ProfilingMetrics& metrics
    );
    
    void estimate_memory_features(
        FeatureVector& fv, 
        const KernelCharacteristics& kc,
        const ProfilingMetrics& metrics
    );
};

} // namespace gridadvisor

#include "feature_extractor_impl.hpp"

#endif
