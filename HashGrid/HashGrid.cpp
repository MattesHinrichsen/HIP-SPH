#include <__clang_hip_runtime_wrapper.h>
#include <cassert>
#include<cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <hip/amd_detail/amd_device_functions.h>
#include <hip/amd_detail/amd_hip_atomic.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/driver_types.h>
#include <stdexcept>
#include<stdlib.h>   
#include <iostream>

#include <hip/hip_runtime.h> 
#include <rocprim/rocprim.hpp>
#include <thrust/detail/copy.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/extrema.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <vector>

#include "HashGrid.h"
#include "HashGrid_Kernels.h"
#include "../Utility/hipHelper.cpp"

#define THREADS_PER_BLOCK 32
#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#ifndef deviceID
    #define deviceID 0
#endif


/*----------------------------------------------------
Construction/Destruction
----------------------------------------------------*/
HashGrid::HashGrid(int n_init, int table_size_init) : n(n_init), table_size(table_size_init) {
    hipSetDevice(deviceID);
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, deviceID);
    std::cout << "HashGrid is using device: " << devProp.name << " with " << devProp.totalGlobalMem / (1024.*1024.*1024.) << " GB of Memory." << std::endl;

    hash_idx = thrust::device_vector<int>(n);
    hash_val = thrust::device_vector<int>(n);

    x = thrust::device_vector<GPU_T>(n);
    y = thrust::device_vector<GPU_T>(n);

    first_index_lookup = thrust::device_vector<int>(table_size_init);
    particles_per_cell = thrust::device_vector<int>(table_size_init);

    n_neighbors = thrust::device_vector<int>(n);
    neighbor_offset = thrust::device_vector<int>(n);
    lst_neighbors = thrust::device_vector<int>(1);

    morton_code = thrust::device_vector<uint32_t>(n);
}

HashGrid::HashGrid() {}

HashGrid::~HashGrid() {

}   

/*----------------------------------------------------
Modify Points
----------------------------------------------------*/
void HashGrid::set_point_from_CPU(int index, GPU_T x_CPU, GPU_T y_CPU) {
    throw std::logic_error("Not implemented");
    // hipMemcpy(&x[index], &x_CPU, sizeof(GPU_T), hipMemcpyHostToDevice);
    // hipMemcpy(thrust::raw_pointer_cast(&y[index]), &y_CPU, sizeof(GPU_T), hipMemcpyHostToDevice);
    // inform_about_change();
}

void HashGrid::set_points_from_CPU(GPU_T* x_CPU, GPU_T* y_CPU) {
    throw std::logic_error("Not implemented");

    // hipMemcpy(x, x_CPU, n*sizeof(GPU_T), hipMemcpyHostToDevice);
    // hipMemcpy(thrust::raw_pointer_cast(&y), y_CPU, n*sizeof(GPU_T), hipMemcpyHostToDevice);
    // inform_about_change();
}

void HashGrid::set_point_from_GPU(int index, GPU_T x_GPU, GPU_T y_GPU) {
    throw std::logic_error("Not implemented");

    // hipMemcpy(&x[index], &x_GPU, sizeof(GPU_T), hipMemcpyHostToDevice);
    // hipMemcpy(&y[index], &y_GPU, sizeof(GPU_T), hipMemcpyHostToDevice);
    // inform_about_change();
}

void HashGrid::set_points_from_GPU(GPU_T* x_GPU, GPU_T* y_GPU) {
    throw std::logic_error("Not implemented");

    // hipMemcpy(x, x_GPU, n*sizeof(GPU_T), hipMemcpyHostToDevice);
    // hipMemcpy((void*)y, y_GPU, n*sizeof(GPU_T), hipMemcpyHostToDevice);
    // inform_about_change();
}

thrust::device_vector<GPU_T>* HashGrid::get_GPU_x() { 
    inform_about_change();
    return &x;
}

thrust::device_vector<GPU_T>* HashGrid::get_GPU_y() { 
    inform_about_change();
    return &y;
}

GPU_T* HashGrid::get_GPU_x_pointer() { 
    inform_about_change();
    return GetPointer(x);
}

GPU_T* HashGrid::get_GPU_y_pointer() { 
    inform_about_change();
    return GetPointer(y);
}

const thrust::device_vector<GPU_T>* const HashGrid::get_const_GPU_x() { 
    return &x;
}

const thrust::device_vector<GPU_T>* const HashGrid::get_const_GPU_y() { 
    return &y;
}

void HashGrid::store_to_file(std::string path) {
    // GPU_T* x_CPU = new GPU_T[n];
    // GPU_T* y_CPU = new GPU_T[n];
    // hipMemcpy(x_CPU, GetPointer(x), n*sizeof(GPU_T), hipMemcpyDeviceToHost);
    // hipMemcpy(y_CPU, GetPointer(y), n*sizeof(GPU_T), hipMemcpyDeviceToHost);
    std::vector<GPU_T> x_CPU(n);
    std::vector<GPU_T> y_CPU(n);
    thrust::copy(x.begin(), x.end(), x_CPU.begin());
    thrust::copy(y.begin(), y.end(), y_CPU.begin());

    std::ofstream file(path, std::ios::binary);

    if (file.is_open()) {
        file.write(reinterpret_cast<char*>(x_CPU.data()), n*sizeof(GPU_T));
        file.write(reinterpret_cast<char*>(y_CPU.data()), n*sizeof(GPU_T));
        file.close();
    } else {
        std::cout << "Couldn't open file!\n";
    }
}

void HashGrid::load_from_file(std::string path) {
    std::vector<GPU_T> x_CPU(n);
    std::vector<GPU_T> y_CPU(n);

    std::ifstream file(path, std::ios::binary);

    if (file.is_open()) {
        file.read(reinterpret_cast<char*>(x_CPU.data()), n*sizeof(GPU_T));
        file.read(reinterpret_cast<char*>(y_CPU.data()), n*sizeof(GPU_T));
        file.close();

        thrust::copy(x_CPU.begin(), x_CPU.end(), x.begin());
        thrust::copy(y_CPU.begin(), y_CPU.end(), y.begin());
    } else {
        std::cout << "Couldn't open file!\n";
    }

    inform_about_change();
}



/*----------------------------------------------------
Debug Info
----------------------------------------------------*/
void HashGrid::print_positions() {
    std::cout << "Positions\n";
    for(int i = 0; i<n; i++) {
        printf("%d: x=%f, y=%f\n", i, x[i], y[i]);
    }
}

void HashGrid::print_hash_map() {
    std::cout << "Hash Map\n";
    for(int i = 0; i<n; i++) {
        std::cout << "idx[" << i << "]=" << hash_idx[i] << ", val[" << i << "]=" << hash_val[i] << "\n";
    }
    std::cout << std::endl;
}

void HashGrid::print_index_lookup() {
    std::cout << "Lookup Table\n";
    for(int i = 0; i<table_size; i++) {
        std::cout << "idx: " << i << ", val: " << first_index_lookup[i] << "\n";
    }
}

void HashGrid::print_particles_per_cell() {
    std::cout << "Particles per Cell\n";
    for(int i = 0; i<table_size; i++) {
        std::cout << "idx: " << i << ", val: " << particles_per_cell[i] << "\n";
    }
}

void HashGrid::print_n_neighbors() {
    std::cout << "n_neighbors\n";
    for(int i = 0; i<n; i++) {
        printf("%d: %d\n", i, n_neighbors[i]);
    }
}

void HashGrid::print_lst_neighbor() {
    std::cout << "lst_neighbors\n";
    for(int i = 0; i<n; i++) {
        printf("%d: %d\n", i, lst_neighbors[i]);
    }
}

void HashGrid::print_neighbor_offset() {
    std::cout << "Neighbor Offset\n";
    for(int i = 0; i<n; i++) {
        printf("%d: %d\n", i, neighbor_offset[i]);
    }
}


/*----------------------------------------------------
Alter internal state
----------------------------------------------------*/
void HashGrid::set_search_radius(GPU_T sr) {
    search_radius = sr;
    inform_about_change();
}

void HashGrid::inform_about_change() {
    isUpdated = false;
    neighborhood_list_is_updated = false;
}

bool HashGrid::is_neighborhood_list_updated() {
    return neighborhood_list_is_updated;
}

int HashGrid::get_total_number_of_neighbors() {
    return total_pairs;
}

int* HashGrid::get_neighbor_list() {
    return GetPointer(lst_neighbors);
}

int* HashGrid::get_number_of_neighbors() {
    return GetPointer(n_neighbors);
}

int* HashGrid::get_neighbor_offset() {
    return GetPointer(neighbor_offset);
}

thrust::device_vector<uint32_t>* HashGrid::z_order_sort() {

    // Find min and max x and y pos
    GPU_T min_x_val = *thrust::min_element(thrust::device, x.begin(), x.end());
    GPU_T max_x_val = *thrust::max_element(thrust::device, x.begin(), x.end()); 
    GPU_T min_y_val = *thrust::min_element(thrust::device, y.begin(), y.end()); 
    GPU_T max_y_val = *thrust::max_element(thrust::device, y.begin(), y.end()); 

    // Compute Morton Code for every point
    hipLaunchKernelGGL( compute_morton_code, 
        (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK,
        std::pow(THREADS_PER_BLOCK, 2),
        0, 0, 
        GetPointer(morton_code),
        n,
        GetPointer(x), GetPointer(y),
        min_x_val, max_x_val,
        min_y_val, max_y_val
        );
    hipDeviceSynchronize();

    // Sort list
    // There might be a more efficient way to do this
    thrust::device_vector<uint32_t> tmp_code(n);
    thrust::copy(thrust::device, morton_code.begin(), morton_code.end(), tmp_code.begin());
    thrust::sort_by_key(thrust::device, tmp_code.begin(), tmp_code.end(), x.begin());
    hipDeviceSynchronize();

    thrust::copy(thrust::device, morton_code.begin(), morton_code.end(), tmp_code.begin());
    thrust::sort_by_key(thrust::device, tmp_code.begin(), tmp_code.end(), y.begin());
    hipDeviceSynchronize();
    
    inform_about_change();
    // Return ordering to SPH
    return &morton_code;
}




/*----------------------------------------------------
Utility functions
----------------------------------------------------*/
void HashGrid::sort_hash_map() {
    // Using: https://rocm.docs.amd.com/projects/rocPRIM/en/latest/device_ops/sort.html#_CPPv4I0000000EN7rocprim16radix_sort_pairsE10hipError_tPvR6size_t17KeysInputIterator18KeysOutputIterator19ValuesInputIterator20ValuesOutputIterator4Sizejj11hipStream_tb

    // size_t temporary_storage_size_bytes;
    // void * temporary_storage_ptr = nullptr;
    // // Get required size of the temporary storage
    // rocprim::radix_sort_pairs(
    //     temporary_storage_ptr, temporary_storage_size_bytes,
    //      , hash_val, hash_idx, hash_idx,
    //     n, 0, ceil(log2(table_size))
    // );
    
    // // allocate temporary storage
    // hipMalloc(&temporary_storage_ptr, temporary_storage_size_bytes);
    
    // // perform sort
    // rocprim::radix_sort_pairs(
    //     temporary_storage_ptr, temporary_storage_size_bytes,
    //     hash_val, hash_val, hash_idx, hash_idx,
    //     n, 0, ceil(log2(table_size))
    // );

    // hipDeviceSynchronize();
    // hipFree(temporary_storage_ptr);

    // thrust::sort_by_key(thrust::device, GetPointer(hash_val), GetPointer(hash_val) + n, GetPointer(hash_idx));
    thrust::sort_by_key(thrust::device, hash_val.begin(), hash_val.end(), hash_idx.begin());
    
    hipDeviceSynchronize();
}   


void HashGrid::update_hash_map() {

    // Just to be sure
    if(isUpdated) return;

    // Compute index of cell and store it in the hash_map 
    hipLaunchKernelGGL( populate_hash_map, 
        (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK,
        std::pow(THREADS_PER_BLOCK, 2),
        0, 0,
        n, 
        GetPointer(x), GetPointer(y), 
        GetPointer(hash_idx), GetPointer(hash_val), 
        search_radius, table_size);
    hipDeviceSynchronize();
    

    // Sort key value pairs
    sort_hash_map();


    // Reset the entries of the first_index_lookup
    hipLaunchKernelGGL( reset_cell_values, 
        (table_size + THREADS_PER_BLOCK)/THREADS_PER_BLOCK,
        std::pow(THREADS_PER_BLOCK, 2),
        0, 0,
        GetPointer(first_index_lookup), GetPointer(particles_per_cell), table_size, n);

    
    // Store the first occurrence of each hash 
    hipLaunchKernelGGL( populate_lookup_and_cell_count, 
        (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK,
        std::pow(THREADS_PER_BLOCK, 2),
        0, 0,
        n, GetPointer(first_index_lookup), GetPointer(particles_per_cell), GetPointer(hash_val));
    hipDeviceSynchronize();
    
    isUpdated = true;
};



/*----------------------------------------------------
Perform Actions on Data
----------------------------------------------------*/
void HashGrid::compute_neighbor_pair_list() {

    if(!isUpdated) update_hash_map();

    // Find how many neighbor pairs there are
    hipLaunchKernelGGL( count_neighbor_pairs, 
        (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK,
        std::pow(THREADS_PER_BLOCK, 2),
        0, 0,
        GetPointer(n_neighbors), n, 
        search_radius, 
        table_size, 
        GetPointer(x), GetPointer(y), 
        GetPointer(first_index_lookup), GetPointer(particles_per_cell), 
        GetPointer(hash_idx));
    hipDeviceSynchronize();

    
    total_pairs = thrust::reduce(thrust::device, n_neighbors.begin(), n_neighbors.end());
    hipDeviceSynchronize();
    
    // Resize the neighbor pair list
    if(total_pairs > lst_neighbors.size()) lst_neighbors.resize(total_pairs);   
    hipDeviceSynchronize();
    

    // Compute prefix sum of neighbor positions
    thrust::exclusive_scan(thrust::device, n_neighbors.begin(), n_neighbors.end(), neighbor_offset.begin());
    hipDeviceSynchronize();

    // Fill neighbor pair list
    hipLaunchKernelGGL( list_neighbor_pairs, 
        (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK,
        std::pow(THREADS_PER_BLOCK, 2),
        0, 0,
        GetPointer(lst_neighbors),
        GetPointer(neighbor_offset),
        total_pairs, 
        GetPointer(n_neighbors),
        n, 
        search_radius, 
        table_size, 
        GetPointer(x), GetPointer(y), 
        GetPointer(first_index_lookup), 
        GetPointer(particles_per_cell), 
        GetPointer(hash_idx));
    hipDeviceSynchronize();

    neighborhood_list_is_updated = true;
}
