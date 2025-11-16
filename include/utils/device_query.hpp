#ifndef DEVICE_QUERY_HPP
#define DEVICE_QUERY_HPP

#include <string>
#include <vector>

namespace gridadvisor {

struct DeviceSpecs {
    // Basic info
    std::string name;
    int device_id;
    
    // Compute capability
    int compute_major;
    int compute_minor;
    
    // Threading
    int sm_count;                    // Number of SMs
    int max_threads_per_sm;          // Max threads per SM
    int max_threads_per_block;       // Max threads per block
    int max_blocks_per_sm;           // Max blocks per SM
    int warp_size;                   // Warp size (32)
    
    // Memory
    size_t global_memory_bytes;      // Total global memory
    size_t shared_mem_per_block;     // Shared memory per block
    size_t shared_mem_per_sm;        // Shared memory per SM
    size_t const_memory_bytes;       // Constant memory
    size_t l2_cache_size;            // L2 cache size
    
    // Registers
    int regs_per_block;              // Max registers per block
    int regs_per_sm;                 // Total registers per SM
    
    // Bandwidth and compute
    float memory_clock_mhz;
    float memory_bandwidth_gb_s;     // Peak memory bandwidth
    int memory_bus_width;
    float clock_rate_mhz;
    float peak_gflops_fp32;          // Estimated peak GFlops
    
    // Methods
    void print() const;
    void to_json(const std::string& filename) const;
};

class DeviceQuery {
public:
    DeviceQuery();
    ~DeviceQuery();
    
    // Query specific device
    DeviceSpecs query(int device_id = 0);
    
    // Get all devices
    std::vector<DeviceSpecs> query_all();
    
    // Get device count
    int get_device_count();
    
private:
    void check_cuda_error(int error, const char* msg);
};

} // namespace gridadvisor

#endif // DEVICE_QUERY_HPP