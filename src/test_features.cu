#include "features/feature_extractor.hpp"
#include <iostream>

// Import kernels from benchmarks
extern __global__ void transpose(const float*, float*, int, int);
extern __global__ void matmul_naive(const float*, const float*, float*, int, int, int);
extern __global__ void matmul_shared(const float*, const float*, float*, int, int, int);
extern __global__ void reduction_sum(const float*, float*, int);

int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "GridAdvisor - Feature Extraction Test\n";
    std::cout << "========================================\n";
    
    gridadvisor::FeatureExtractor extractor(0);
    
    // Test 1: Transpose (memory-bound)
    std::cout << "\n--- Testing: Transpose ---\n";
    auto fv1 = extractor.extract(transpose, 2048 * 2048);
    fv1.print();
    
    // Test 2: Matrix multiply naive (compute-bound)
    std::cout << "\n--- Testing: Matrix Multiply Naive ---\n";
    auto fv2 = extractor.extract(matmul_naive, 512 * 512);
    fv2.print();
    
    // Test 3: Matrix multiply shared (compute-bound with shared mem)
    std::cout << "\n--- Testing: Matrix Multiply Shared ---\n";
    auto fv3 = extractor.extract(matmul_shared, 512 * 512);
    fv3.print();
    
    // Test 4: Reduction (memory-bound with shared mem)
    std::cout << "\n--- Testing: Reduction ---\n";
    auto fv4 = extractor.extract(reduction_sum, 1 << 24);
    fv4.print();
    
    // Show feature arrays
    std::cout << "\n=== Feature Arrays ===\n\n";
    
    std::cout << "Transpose features:\n[";
    auto arr1 = fv1.to_array();
    for (size_t i = 0; i < arr1.size(); i++) {
        std::cout << arr1[i];
        if (i < arr1.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n\n";
    
    std::cout << "MatMul Shared features:\n[";
    auto arr3 = fv3.to_array();
    for (size_t i = 0; i < arr3.size(); i++) {
        std::cout << arr3[i];
        if (i < arr3.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n\n";
    
    std::cout << "========================================\n";
    std::cout << "✓ Feature extraction test completed!\n";
    std::cout << "========================================\n\n";
    
    return 0;
}