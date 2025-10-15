#pragma once
#include <cstdint>
#include <string>
#include <thrust/device_vector.h>
#include <vector>
#include "../Utility/type.h"

class HashGrid {
    int n;                                      // Number of Objects stored in data-structure
    thrust::device_vector<GPU_T> x;                               // x-coordinate of the object
    thrust::device_vector<GPU_T> y;                           // y-coordinate of the object

    int table_size;                             // Numbers of entries in the hash map
    thrust::device_vector<int> first_index_lookup;                  // Hash map: Stores the index into hash_map of the first element with the hash of the cell
    thrust::device_vector<int> hash_idx;        
    thrust::device_vector<int> hash_val;
    thrust::device_vector<int> particles_per_cell;

    thrust::device_vector<int> n_neighbors;       // For each point stores the number of neighboring points
    thrust::device_vector<int> neighbor_offset;
    thrust::device_vector<int> lst_neighbors;     // Stores all neighborhood pairs (Big list). Has to be cast to thrust::device_vector for use.
    int total_pairs = 0;

    thrust::device_vector<uint32_t> morton_code;

    GPU_T search_radius;                            // Radius in which neighboring points are searched

    bool isUpdated = false;                         // Track wether changes have been made to the points
    bool neighborhood_list_is_updated = false;

    // Utility functions
    void sort_hash_map();
    void update_hash_map();

public:
    HashGrid(int, int);
    HashGrid();
    ~HashGrid();

    // Functions that alter internal states
    void set_search_radius(GPU_T);
    void inform_about_change();

    // Functions to modify points
    void load_from_file(std::string path);

    void set_point_from_GPU(int index, GPU_T x_GPU, GPU_T y_GPU);
    void set_points_from_GPU(GPU_T* x_GPU, GPU_T* y_GPU);

    void set_point_from_CPU(int index, GPU_T x_CPU, GPU_T y_CPU);
    void set_points_from_CPU(GPU_T* x_CPU, GPU_T* y_CPU);

    thrust::device_vector<GPU_T>* get_GPU_x();
    thrust::device_vector<GPU_T>* get_GPU_y();
    GPU_T* get_GPU_x_pointer();
    GPU_T* get_GPU_y_pointer();


    // Interface with neighborhood list
    int get_total_number_of_neighbors();
    bool is_neighborhood_list_updated();

    int* get_neighbor_list();
    int* get_number_of_neighbors();
    int* get_neighbor_offset();

    thrust::device_vector<uint32_t>* z_order_sort();

    // Functions that do not alter internal state
    const thrust::device_vector<GPU_T>* const get_const_GPU_x();
    const thrust::device_vector<GPU_T>* const get_const_GPU_y();

    
    void store_to_file(std::string path);
    
    // Debuging
    void print_positions();
    void print_hash_map();
    void print_index_lookup();
    void print_particles_per_cell();
    void print_n_neighbors();
    void print_lst_neighbor();
    void print_neighbor_offset();


    // Perform actions on the data
    void compute_neighbor_pair_list();
};