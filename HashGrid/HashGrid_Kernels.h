#include "HashGrid.h"
#include <__clang_hip_runtime_wrapper.h>
#include <cassert>
#include<cmath>
#include <cstdio>
#include <hip/amd_detail/amd_hip_atomic.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include<stdlib.h>   

#include <hip/hip_runtime.h> 
#include <rocprim/rocprim.hpp>
#include <sys/types.h>
#include <thrust/sort.h>


__device__ void hash(int* out, const int x,const int y, const int table_size);

__global__ void populate_hash_map(int n, GPU_T* x, GPU_T* y, int* hash_idx, int* hash_val, const GPU_T search_radius, const int table_size);

__global__ void reset_cell_values(int* first_index_lookup, int* particles_per_cell, const int table_size, const int n);

__global__ void populate_lookup_and_cell_count(const int n, int* first_index_lookup, int* particles_per_cell, const int* const hash_val);

__global__ void count_neighbor_pairs(int* out, const int n, const GPU_T search_radius, const int table_size, const GPU_T* x, const GPU_T* y, const int* first_index_lookup, const int* particles_per_cell, const int* hash_idx);

__global__ void list_neighbor_pairs(int* lst_neighbors, int* neighbor_offset, int total_number_pairs, int* n_neighbors, const int n, const GPU_T search_radius, const int table_size, const GPU_T* x, const GPU_T* y, const int* first_index_lookup, const int* particles_per_cell, const int* hash_idx);

__global__ void compute_morton_code(u_int32_t* out, int n, GPU_T* x, GPU_T* y, GPU_T min_x, GPU_T max_x, GPU_T min_y, GPU_T max_y);