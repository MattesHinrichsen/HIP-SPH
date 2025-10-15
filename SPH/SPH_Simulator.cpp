#pragma once

#include "SPH_Simulator.h"
#include "SPH_Kernels.cpp"
#include "../Utility/type.h"
#include "../Utility/hipHelper.cpp"

#include <__clang_hip_runtime_wrapper.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <format>
#include <hip/amd_detail/amd_hip_atomic.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h> 
#include <iomanip>
#include <rocprim/rocprim.hpp>
#include <rocrand/rocrand.h>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

#define THREADS_PER_BLOCK 32
#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#ifndef deviceID
    #define deviceID 0
#endif


// CPU functions
SPH_Simulator::SPH_Simulator(int N_points, int max_x_coord, int max_y_coord) : n(N_points) {

    vx = thrust::device_vector<GPU_T>(n);
    vy = thrust::device_vector<GPU_T>(n);
    density = thrust::device_vector<GPU_T>(n);
    shepard_correction_term = thrust::device_vector<GPU_T>(n);
    force_x = thrust::device_vector<GPU_T>(n);
    force_y = thrust::device_vector<GPU_T>(n);

    //TODO: Variable table size
    // If this is placed before the above malloc weird things happen!
    max_x = max_x_coord;
    max_y = max_y_coord;
    GPU_T radius = 2.5*std::sqrt(max_x*max_y / (GPU_T)n);

    mass = 30.0 * (max_x * max_y) / n;

    int table_size = max(10, 1.2 * (max_x * max_y) / (radius * radius));
    hg = HashGrid(n, table_size);

    Set_Smoothing_Radius(radius);
    Set_Push_Pull_Radius(smoothing_radius);
   
    hipLaunchKernelGGL( initialize_velocity, 
        (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK,
        std::pow(THREADS_PER_BLOCK, 2),
        0, 0,
        GetPointer(vx), GetPointer(vy), n);
    hipDeviceSynchronize();

    x_CPU = std::vector<GPU_T>(n);
    y_CPU = std::vector<GPU_T>(n);
    vx_CPU = std::vector<GPU_T>(n);
    vy_CPU = std::vector<GPU_T>(n);
}

SPH_Simulator::~SPH_Simulator() {

}

void SPH_Simulator::Simulation_Step() {

    if (!hg.is_neighborhood_list_updated()) hg.compute_neighbor_pair_list();

    Calculate_Density();

    Reset_Forces();

    Calculate_Pressure_and_Viscosity_Force();

    Calculate_Mouse_Pointer_Forces();
    Calculate_Gravity_Forces();

    Time_Integration();

    Resolve_Collision();

    time_step++;

    // Print_Positions();
    // Print_Densities();
    // Print_Forces();
    // Print_Velocities();

}

GPU_T SPH_Simulator::Get_Scale(GPU_T scaling_factor) {
    throw std::logic_error("Not implemented");
    return (GPU_T)std::max(max_x, max_y) / (300*std::sqrt(n)) * scaling_factor;
}

void SPH_Simulator::Randomize_Points() {
    // Points are always stored between 0 and 1
    // Only for visualization they will be projected to the range [0, max_x/max_y]
    

    auto x = hg.get_GPU_x_pointer();
    auto y = hg.get_GPU_y_pointer();

    rocrand_generator generator;
    rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT);
    rocrand_initialize_generator(generator);

    rocrand_generate_uniform(generator, x, n);
    rocrand_generate_uniform(generator, y, n);
    hipDeviceSynchronize();

    hipLaunchKernelGGL( scale_random_points_to_resolution, 
        (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK,
        std::pow(THREADS_PER_BLOCK, 2),
        0, 0,
        x, y, 
        n, max_x, max_y);
    hipDeviceSynchronize();
}

void SPH_Simulator::Grid_Points() {

    int points_per_column = ceil(std::sqrt(n));
    GPU_T side_length = std::min(max_x, max_y) * 0.8;

    GPU_T x_start = 0.5 * max_x - side_length / (2.);
    GPU_T y_start = 0.5 * max_y - side_length / (2.);
    GPU_T dx = side_length / (points_per_column - 1); 
    GPU_T dy = side_length / (points_per_column - 1); 

    auto x = hg.get_GPU_x_pointer();
    auto y = hg.get_GPU_y_pointer();

    hipLaunchKernelGGL( initialize_particles_in_grid, 
        dim3((points_per_column + THREADS_PER_BLOCK)/THREADS_PER_BLOCK, (points_per_column + THREADS_PER_BLOCK)/THREADS_PER_BLOCK),
        dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK),
        0, 0,
        x, y,
        n, points_per_column, x_start, y_start, dx, dy);
    hipDeviceSynchronize();

}

void SPH_Simulator::Time_Integration() {
    auto x = hg.get_GPU_x_pointer();
    auto y = hg.get_GPU_y_pointer();

    hipLaunchKernelGGL( time_integration_kernel, 
        (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK,
        std::pow(THREADS_PER_BLOCK, 2),
        0, 0,
        x, 
        y, 
        GetPointer(vx), 
        GetPointer(vy), 
        GetPointer(force_x), 
        GetPointer(force_y), 
        GetPointer(density), 
        n, dt, time_step);
    hipDeviceSynchronize();
}

void SPH_Simulator::Calculate_Density() {

    hipLaunchKernelGGL( reset_densities, 
        (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK,
        std::pow(THREADS_PER_BLOCK, 2),
        0, 0,
        GetPointer(density), 
        GetPointer(shepard_correction_term), 
        n);
    hipDeviceSynchronize();
    

    hipLaunchKernelGGL( run_density_kernels, 
        (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK,
        std::pow(THREADS_PER_BLOCK, 2),
        0, 0,
        n, 
        hg.get_number_of_neighbors(), 
        hg.get_neighbor_list(), 
        hg.get_neighbor_offset(), 
        hg.get_GPU_x_pointer(), 
        hg.get_GPU_y_pointer(),
        GetPointer(density), 
        GetPointer(shepard_correction_term),
        smoothing_radius, 
        mass, 
        density_factor);
    hipDeviceSynchronize();

    // hipLaunchKernelGGL( apply_shepard_correction, 
    //     (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK,
    //     std::pow(THREADS_PER_BLOCK, 2),
    //     0, 0,
    //     n,
    //     GetPointer(density), 
    //     GetPointer(shepard_correction_term)
    //     );
    // hipDeviceSynchronize();

}

GPU_T SPH_Simulator::Calculate_Density_At_Point(GPU_T x_pos, GPU_T y_pos) {

    // throw std::logic_error("Not implemented");
    auto x = hg.get_GPU_x_pointer();
    auto y = hg.get_GPU_y_pointer();

    GPU_T* density_GPU;
    hipMalloc(&density_GPU, sizeof(GPU_T));
    
    hipLaunchKernelGGL( density_at_point_kernel, 
        1,
        1,
        0, 0,
        density_GPU, x, y, x_pos , y_pos , n, smoothing_radius, density_factor, mass);
    hipDeviceSynchronize();

    GPU_T density_CPU;
    hipMemcpy(&density_CPU, density_GPU, sizeof(GPU_T), hipMemcpyDeviceToHost);
    hipFree(density_GPU);

    return density_CPU;
}

void SPH_Simulator::Resolve_Collision() {
    
    auto x = hg.get_GPU_x_pointer();
    auto y = hg.get_GPU_y_pointer();

    hipLaunchKernelGGL( resolve_collision_kernel, 
        (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK,
        std::pow(THREADS_PER_BLOCK, 2),
        0, 0,
        x, y, 
        GetPointer(vx), GetPointer(vy), 
        n, max_x, 
        max_y, collisionDamping);
    hipDeviceSynchronize();
}

void SPH_Simulator::Reset_Forces() {
    hipLaunchKernelGGL( reset_forces, 
        (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK,
        std::pow(THREADS_PER_BLOCK, 2),
        0, 0,
        GetPointer(force_x), 
        GetPointer(force_y), 
        n);
    hipDeviceSynchronize();

}

void SPH_Simulator::Set_Pull_Push_Mode(int mode, GPU_T x_pos, GPU_T y_pos) {
    // 0: Do nothing
    // 1: Pull
    // 2: Push
    push_pull_mode = mode;
    push_pull_x = x_pos;
    push_pull_y = y_pos;
}

void SPH_Simulator::Calculate_Pressure_and_Viscosity_Force() {
    hipLaunchKernelGGL( run_force_kernels, 
        (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK,
        std::pow(THREADS_PER_BLOCK, 2),
        0, 0,
        n, 
        hg.get_number_of_neighbors(), 
        hg.get_neighbor_list(), 
        hg.get_neighbor_offset(), 
        hg.get_GPU_x_pointer(), 
        hg.get_GPU_y_pointer(), 
        GetPointer(vx), 
        GetPointer(vy), 
        GetPointer(force_x), 
        GetPointer(force_y), 
        GetPointer(density), 
        smoothing_radius, 
        mass, 
        density_factor, 
        pressureMultiplier, 
        targetDensity, 
        dynamic_viscosity);
    hipDeviceSynchronize();

}


void SPH_Simulator::Calculate_Gravity_Forces() {
    hipLaunchKernelGGL( gravitational_force_kernel, 
        (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK,
        std::pow(THREADS_PER_BLOCK, 2),
        0, 0,
        GetPointer(force_y), 
        mass,
        n);
    hipDeviceSynchronize();
}

void SPH_Simulator::Calculate_Mouse_Pointer_Forces() {

    auto x = hg.get_GPU_x_pointer();
    auto y = hg.get_GPU_y_pointer();

    if(push_pull_mode != 0) {
        hipLaunchKernelGGL( mouse_pointer_force_kernel, 
            (n + THREADS_PER_BLOCK)/THREADS_PER_BLOCK,
            std::pow(THREADS_PER_BLOCK, 2),
            0, 0,
            x, 
            y, 
            GetPointer(force_x), 
            GetPointer(force_y), 
            n, 
            mass,
            push_pull_x , push_pull_y , 
            push_pull_radius, 
            push_pull_mode);
        hipDeviceSynchronize();

        push_pull_mode = 0;
    }
}

void SPH_Simulator::Set_Push_Pull_Radius(GPU_T radius) {
    push_pull_radius = radius;
}

void SPH_Simulator::Set_Smoothing_Radius(GPU_T radius) {
    smoothing_radius = radius ;
    hg.set_search_radius(radius);

    density_factor = 6 / (PI * std::pow(smoothing_radius,4));
}

GPU_T SPH_Simulator::Get_Smoothing_Radius() {
    return smoothing_radius;
}


std::vector<GPU_T>& SPH_Simulator::get_x() { 
    auto x = hg.get_GPU_x();
    thrust::copy(x->begin(), x->end(), x_CPU.begin());
    return x_CPU; 
}

std::vector<GPU_T>& SPH_Simulator::get_y() { 
    auto y = hg.get_GPU_y();
    thrust::copy(y->begin(), y->end(), y_CPU.begin());
    return y_CPU; 
}

std::vector<GPU_T>& SPH_Simulator::get_vx() { 
    thrust::copy(vx.begin(), vx.end(), vx_CPU.begin());
    return vx_CPU; 
}

std::vector<GPU_T>& SPH_Simulator::get_vy() { 
    thrust::copy(vy.begin(), vy.end(), vy_CPU.begin());
    return vy_CPU; 
}

void SPH_Simulator::store_to_file(std::string path) {
    hg.store_to_file(path);
}

void SPH_Simulator::load_from_file(std::string path) {
    hg.load_from_file(path);
}

void SPH_Simulator::Set_Target_Density(GPU_T target) {
    targetDensity = target; 
}

void SPH_Simulator::Set_Pressure_Multiplier(GPU_T multiplier) {
    pressureMultiplier = multiplier;
}

void SPH_Simulator::Print_Positions() {
    hg.print_positions();
}

void SPH_Simulator::Print_Forces() {
    printf("Forces\n");
    for(int j = 0; j<n; j++) {
        printf("%d: %f %f\n", j, force_x[j], force_y[j]);
    }
}

void SPH_Simulator::Print_Densities() {
    printf("Density\n");
    for(int j = 0; j<n; j++) {
        printf("%d: %.16f\n", j, density[j]);
    }
}

void SPH_Simulator::Print_Velocities() {
    printf("Velocities\n");
    for(int j = 0; j<n; j++) {
        printf("%d: %f %f\n", j, vx[j], vy[j]);
    }
}


void SPH_Simulator::z_order_sort() {
    performing_z_order_sort = true;
    auto morton_code = hg.z_order_sort();

    // Sort velocity
    // There might be a more efficient way to do this
    thrust::device_vector<uint32_t> tmp_code(n);
    thrust::copy(thrust::device, morton_code->begin(), morton_code->end(), tmp_code.begin());
    thrust::sort_by_key(thrust::device, tmp_code.begin(), tmp_code.end(), vx.begin());
    hipDeviceSynchronize();

    thrust::copy(thrust::device, morton_code->begin(), morton_code->end(), tmp_code.begin());
    thrust::sort_by_key(thrust::device, tmp_code.begin(), tmp_code.end(), vy.begin());
    hipDeviceSynchronize();

    performing_z_order_sort = false;
}
