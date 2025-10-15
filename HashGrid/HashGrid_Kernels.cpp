#include "HashGrid.h"
#include <__clang_hip_runtime_wrapper.h>
#include <cassert>
#include<cmath>
#include <cstdint>
#include <cstdio>
#include <hip/amd_detail/amd_hip_atomic.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include<stdlib.h>   

#include <hip/hip_runtime.h> 
#include <rocprim/rocprim.hpp>
#include <thrust/sort.h>
#include "HashGrid_Kernels.h"
#include "Z_Order.cpp"



__device__ void hash(int* out, const int x,const int y, const int table_size) {
    constexpr int prime = 1009;
    *out = abs((prime + x) * prime + y) % table_size;
}

__global__ void populate_hash_map(int n, GPU_T* x, GPU_T* y, int* hash_idx, int* hash_val, const GPU_T search_radius, const int table_size) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if(i<n) {
        int xi = std::floor(x[i] / search_radius);
        int yi = std::floor(y[i] / search_radius);
        
        hash_idx[i] = i;
        hash(&hash_val[i], xi, yi, table_size);
    }
}

__global__ void reset_cell_values(int* first_index_lookup, int* particles_per_cell, const int table_size, const int n) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if(i<table_size) {

        particles_per_cell[i] = 0;
        first_index_lookup[i] = n+1;
    }
}

__global__ void populate_lookup_and_cell_count(const int n, int* first_index_lookup, int* particles_per_cell, const int* const hash_val) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < n) {
        int idx = hash_val[i];
        atomicAdd(&particles_per_cell[idx],1);
        atomicMin(&first_index_lookup[idx], i);
    }
}


__global__ void count_neighbor_pairs(int* out, const int n, const GPU_T search_radius, const int table_size, const GPU_T* x, const GPU_T* y, const int* first_index_lookup, const int* particles_per_cell, const int* hash_idx) {
    int index = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if (index < n) {
        int neighbor_count = 0;

        // Compute hash of current and all neighboring cells
        int hash_list[9];
        int xi = std::floor(x[index] / search_radius);
        int yi = std::floor(y[index] / search_radius);
        for(int x_offset = -1; x_offset<2; x_offset++) {
            for(int y_offset = -1; y_offset<2; y_offset++) {
                hash(&hash_list[(x_offset + 1) * 3 + y_offset + 1], xi + x_offset, yi + y_offset, table_size);
            }
        }

        // Make list unique
        for(int i = 0; i < 9; i++) {
            for(int j = i+1; j < 9; j++) {
                if (hash_list[i] == hash_list[j]) hash_list[j] = -1;
            }
        }

        // Loop over all relevant cells
        for(int h_idx = 0; h_idx < 9; h_idx++) {
            if (hash_list[h_idx] != -1) {
                // Look up index of first element of current hash
                int first_index_of_current_cell = first_index_lookup[hash_list[h_idx]];
                int end_iter_index = first_index_of_current_cell + particles_per_cell[hash_list[h_idx]];
                for(int iter_idx = first_index_of_current_cell; iter_idx < end_iter_index; iter_idx++){
                    
                    // Get the index into x corresponding to the current iter_idx of the other point
                    int x_y_idx = hash_idx[iter_idx];

                    // If we are not looking at ourself
                    if (x_y_idx > index) {
                        // Check if other point is truly within the search radius
                        GPU_T R = (x[index] - x[x_y_idx]) * (x[index] - x[x_y_idx]) + (y[index] - y[x_y_idx]) * (y[index] - y[x_y_idx]);
                        if (R <= search_radius * search_radius) {
                            // We found a neighbor
                            neighbor_count++;
                        } 
                    }
                }
            }
            
        }
        out[index] = neighbor_count;
    }
}


__global__ void list_neighbor_pairs(int* lst_neighbors, int* neighbor_offset, int total_number_pairs, int* n_neighbors, const int n, const GPU_T search_radius, const int table_size, const GPU_T* x, const GPU_T* y, const int* first_index_lookup, const int* particles_per_cell, const int* hash_idx) {
    int index = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    
    if (index < n) {
        int neighbor_count = 0;

        // Compute hash of current and all neighboring cells
        int hash_list[9];

        int xi = std::floor(x[index] / search_radius);
        int yi = std::floor(y[index] / search_radius);
        for(int x_offset = -1; x_offset<2; x_offset++) {
            for(int y_offset = -1; y_offset<2; y_offset++) {
                hash(&hash_list[(x_offset + 1) * 3 + y_offset + 1], xi + x_offset, yi + y_offset, table_size);
            }
        }

        // Make list unique
        for(int i = 0; i < 9; i++) {
            for(int j = i+1; j < 9; j++) {
                if (hash_list[i] == hash_list[j]) hash_list[j] = -1;
            }
        }


        
        // Loop over all relevant cells
        for(int h_idx = 0; h_idx < 9; h_idx++) {
            if (hash_list[h_idx] != -1) {
                // Look up index of first element of current hash

                int first_index_of_current_cell = first_index_lookup[hash_list[h_idx]];

                int end_iter_index = first_index_of_current_cell + particles_per_cell[hash_list[h_idx]];
                for(int iter_idx = first_index_of_current_cell; iter_idx < end_iter_index; iter_idx++){
                    
                    // Get the index into x corresponding to the current iter_idx of the other point
                    int x_y_idx = hash_idx[iter_idx];
                    
                    // If we are not looking at ourself
                    if (x_y_idx > index) {
                        // Check if other point is truly within the search radius

                        GPU_T R = (x[index] - x[x_y_idx]) * (x[index] - x[x_y_idx]) + (y[index] - y[x_y_idx]) * (y[index] - y[x_y_idx]);

                        if (R <= search_radius * search_radius) {
                            // We found a neighbor
                            lst_neighbors[neighbor_offset[index] + neighbor_count] = x_y_idx;
                            neighbor_count++;
                        } 
                    }
                }
            }
            
        } 
    }
}


__global__ void compute_morton_code(uint32_t* out, int n, GPU_T* x, GPU_T* y, GPU_T min_x, GPU_T max_x, GPU_T min_y, GPU_T max_y) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    
    if (i < n) {    

        uint32_t xi = float_to_int(x[i], min_x, max_x);
        uint32_t yi = float_to_int(y[i], min_y, max_y);
        out[i] = morton_code(xi, yi);
    }
}