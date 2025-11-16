#include "utils/device_query.hpp"
#include <iostream>
#include <exception>

int main() {
    try {
        std::cout << "GridAdvisor - Device Query Tool\n";
        std::cout << "================================\n";
        
        gridadvisor::DeviceQuery query;
        
        int device_count = query.get_device_count();
        std::cout << "\nFound " << device_count << " CUDA device(s)\n";
        
        // Query all devices
        auto all_devices = query.query_all();
        
        for (const auto& device : all_devices) {
            device.print();
            
            // Save to JSON
            std::string json_file = "device_" + std::to_string(device.device_id) + ".json";
            device.to_json(json_file);
            std::cout << "Device specs saved to: " << json_file << "\n";
        }
        
        std::cout << "\n✓ Device query completed successfully!\n\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Error: " << e.what() << "\n\n";
        return 1;
    }
}
