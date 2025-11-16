#ifndef FEATURE_EXTRACTOR_HPP
#define FEATURE_EXTRACTOR_HPP

#include "utils/device_query.hpp"
#include <cuda.h>
#include <string>
#include <vector>

namespace gridadvisor {

// Our 24-dimensional feature vector
struct FeatureVector {
    // Hardware features (5)
    float f1_sm_count;
    float f2_max_threads_per_sm;
    float f3_memory_bandwidth_gb_s;
    float f4_peak_gflops;
    float f5_shared_memory_per_sm_kb;
    
    // Kernel resource features (6)
    float f6_registers_per_thread;
    float f7_shared_memory_per_block_kb;
    float f8_local_memory_per_thread_b;
    float f9_const_memory_kb;
    float f10_instruction_count;  // Estimated
    float f11_binary_size_kb;
    
    // Compute characteristics (5) - Estimated based on kernel type
    float f12_arithmetic_intensity;
    float f13_flop_count;
    float f14_memory_ops;
    float f15_fp_ratio;
    float f16_transcendental_ratio;
    
    // Memory access features (4) - Heuristics
    float f17_access_pattern_score;  // 0=random, 1=coalesced
    float f18_shared_mem_reuse;
    float f19_global_load_ratio;
    float f20_memory_divergence;
    
    // Control flow features (2)
    float f21_branch_intensity;
    float f22_sync_intensity;
    
    // Problem size features (2)
    float f23_total_work_items;
    float f24_work_per_sm;
    
    void print() const;
    std::vector<float> to_array() const;
};

// Kernel characteristics that we can extract
struct KernelCharacteristics {
    std::string name;
    int registers_per_thread;
    int shared_memory_static;
    int shared_memory_dynamic;
    int const_memory;
    int local_memory;
    int max_threads_per_block;
    
    // Estimated characteristics (based on kernel type)
    enum class Type {
        MEMORY_BOUND,
        COMPUTE_BOUND,
        BALANCED,
        UNKNOWN
    } type;
    
    bool uses_shared_memory;
    bool has_atomics;
    float estimated_ai;  // Arithmetic intensity estimate
};

class FeatureExtractor {
public:
    FeatureExtractor(int device_id = 0);
    
    // Extract features from a kernel function
    template<typename KernelFunc>
    FeatureVector extract(KernelFunc kernel, int problem_size);
    
    // Extract just kernel characteristics
    template<typename KernelFunc>
    KernelCharacteristics get_kernel_characteristics(KernelFunc kernel);
    
private:
    int device_id;
    DeviceSpecs device_specs;
    
    void estimate_compute_features(FeatureVector& fv, 
                                   const KernelCharacteristics& kc);
    void estimate_memory_features(FeatureVector& fv, 
                                  const KernelCharacteristics& kc);
};

} // namespace gridadvisor

#include "feature_extractor_impl.hpp"

#endif
