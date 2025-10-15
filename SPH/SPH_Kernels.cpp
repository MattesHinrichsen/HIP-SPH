#include <__clang_hip_runtime_wrapper.h>
#include <cassert>
#include<cmath>
#include <cstdio>
#include <hip/amd_detail/amd_hip_atomic.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include<stdlib.h>   

#include <hip/hip_runtime.h> 
#include <rocprim/rocprim.hpp>
#include <thrust/sort.h>

#include "../Utility/type.h"


__device__ void Smoothing_Kernel(GPU_T* out, GPU_T dst, GPU_T smoothing_radius, GPU_T density_factor) {
    GPU_T value = smoothing_radius - dst;
    *out = value * value * density_factor;
}

__device__ void Convert_Density_GPU_To_Pressure(GPU_T* out, GPU_T density, GPU_T pressureMultiplier, GPU_T targetDensity) {
    *out = pressureMultiplier * (density - targetDensity);
}

__global__ void initialize_velocity(GPU_T* vx, GPU_T* vy, int n) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if(i<n) {
        vx[i] = 0;
        vy[i] = 0;
    }
}

__global__ void scale_random_points_to_resolution(GPU_T* x, GPU_T* y, int n, GPU_T x_scale, GPU_T y_scale) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if(i<n) {
        x[i] *= x_scale;
        y[i] *= y_scale;
    }
}   

__global__ void initialize_particles_in_grid(GPU_T* x, GPU_T* y, int n, int points_per_column, GPU_T x_start, GPU_T y_start, GPU_T dx, GPU_T dy) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int j = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    if(i < points_per_column and j < points_per_column and points_per_column * i + j < n) {
        x[points_per_column * i + j] = x_start + dx * i;
        y[points_per_column * i + j] = y_start + dy * j;
    }
}

__global__ void reset_densities(GPU_T* density, GPU_T* shepard_correction_term, int n) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if(i<n) {
        density[i] = 0;
        shepard_correction_term[i] = 0;
    }
}

__global__ void reset_forces(GPU_T* force_x, GPU_T* force_y, int n) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if(i<n) {
        force_x[i] = 0;
        force_y[i] = 0;
    }
}


inline __device__ void density_kernel(int i, int j, GPU_T dst, GPU_T* density, GPU_T* shepard_correction_term, GPU_T smoothing_radius, GPU_T mass, GPU_T density_factor){  

    GPU_T influence = 10;
    Smoothing_Kernel(&influence, dst, smoothing_radius, density_factor);

    atomicAdd(&density[i], mass * influence);
    atomicAdd(&shepard_correction_term[i], influence);
    if(i != j) {
        atomicAdd(&density[j], mass * influence);
        atomicAdd(&shepard_correction_term[j], influence);
    } 
}

__global__ void density_at_point_kernel(GPU_T* out, const GPU_T* x, const GPU_T* y, const GPU_T x_pos, const GPU_T y_pos, const int n, const GPU_T smoothing_radius, const GPU_T density_factor, const GPU_T mass) {
    *out = 0;
    for(int i = 0; i<n; i++) {
        GPU_T dst = std::sqrt(std::pow(x_pos - x[i], 2) + std::pow(y_pos - y[i], 2));

        if(dst  > smoothing_radius) continue;

        GPU_T influence;
        Smoothing_Kernel(&influence, dst, smoothing_radius, density_factor);
        *out += mass * influence;
    }
}

__global__ void resolve_collision_kernel(GPU_T* x, GPU_T* y, GPU_T* vx, GPU_T* vy, int n, GPU_T max_x, GPU_T max_y, GPU_T collisionDamping) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

    if (i < n) {
        if(x[i] > max_x) {
            x[i] = max_x;
            vx[i] *= -1 * collisionDamping;
        }
        if(x[i] < 0) {
            x[i] = 0;
            vx[i] *= -1 * collisionDamping;
        }
        if(y[i] > max_y) {
            y[i] = max_y;
            vy[i] *= -1 * collisionDamping;
        }
        if(y[i] < 0) {
            y[i] = 0;
            vy[i] *= -1 * collisionDamping;
        }
    }
}

inline __device__ void pressure_force_kernel(int i, int j, GPU_T dst, GPU_T* x, GPU_T* y, GPU_T* force_x, GPU_T* force_y, GPU_T* density, GPU_T smoothing_radius, GPU_T mass, GPU_T density_factor, GPU_T pressureMultiplier, GPU_T targetDensity) {
    GPU_T pressure_i; 
    GPU_T pressure_j;
    Convert_Density_GPU_To_Pressure(&pressure_i, density[i], pressureMultiplier, targetDensity);
    Convert_Density_GPU_To_Pressure(&pressure_j, density[j], pressureMultiplier, targetDensity);

    GPU_T fxi;
    GPU_T fxj;
    GPU_T fyi;
    GPU_T fyj;

    if (dst > 1e-8) {
        GPU_T factor = 2 * (smoothing_radius - dst) / dst  
                     * (pressure_i + pressure_j) / 2. * mass * density_factor;

        fxi =  factor * (x[i] - x[j]) / density[j];  
        fyi =  factor * (y[i] - y[j]) / density[j]; 
        fxj = -factor * (x[i] - x[j]) / density[i];  
        fyj = -factor * (y[i] - y[j]) / density[i]; 
    } else {
        GPU_T factor = 2 * (smoothing_radius - dst) 
                     * (pressure_i + pressure_j) / 2. * mass * density_factor;

        fxi =  factor / density[j];  
        fyi =  factor / density[j]; 
        fxj = -factor / density[i];  
        fyj = -factor / density[i]; 
        
    }
    atomicAdd(&force_x[i], fxi);
    atomicAdd(&force_y[i], fyi);
    atomicAdd(&force_x[j], fxj);
    atomicAdd(&force_y[j], fyj);
}

inline __device__ void viscous_force_kernel(int i, int j, GPU_T* x, GPU_T* y, GPU_T* vx, GPU_T* vy, GPU_T* force_x, GPU_T* force_y, GPU_T* density, GPU_T smoothing_radius, GPU_T mass, GPU_T dynamic_viscosity, GPU_T density_factor) {
    GPU_T inside_sqrt = x[i]*x[i] + x[j]*x[j] + y[i]*y[i] + y[j]*y[j] - 2*x[i]*x[j] - 2*y[i]*y[j];
    GPU_T factor;

    if (inside_sqrt > 1e-6) {
        factor =  (-2*smoothing_radius / std::sqrt(inside_sqrt) + 4) * mass * dynamic_viscosity * density_factor;
        factor = std::max(factor, GPU_T(0));
        // atomicMax(&factor,0);
    } else {
        factor =  0;
    }

    atomicAdd(&force_x[i],  factor * (vx[j] - vx[i]) / density[j]);
    atomicAdd(&force_y[i],  factor * (vy[j] - vy[i]) / density[j]);
    atomicAdd(&force_x[j], -factor * (vx[j] - vx[i]) / density[i]);
    atomicAdd(&force_y[j], -factor * (vy[j] - vy[i]) / density[i]);
}

__global__ void gravitational_force_kernel(GPU_T* force_y, GPU_T mass, int n) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < n) {
        force_y[i] -= 10.68 * mass;//0.05;
    }
}

__global__ void mouse_pointer_force_kernel(const GPU_T* x, const GPU_T* y, GPU_T* force_x, GPU_T* force_y, int n, GPU_T mass, GPU_T push_pull_x, GPU_T push_pull_y, GPU_T radius, int push_pull_mode) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    GPU_T pull_factor = 2000;
    if (i < n) {
        GPU_T dst = std::sqrt(std::pow(x[i] - push_pull_x, 2) + std::pow(y[i] - push_pull_y, 2));
        if(dst  <= radius) {
            if (push_pull_mode == 1){
                force_x[i] += (push_pull_x - x[i]) / dst * pull_factor * mass;
                force_y[i] += (push_pull_y - y[i]) / dst * pull_factor * mass;
            } else if(push_pull_mode == 2) {
                force_x[i] -= (push_pull_x - x[i]) / dst * pull_factor * mass;
                force_y[i] -= (push_pull_y - y[i]) / dst * pull_factor * mass;
            }   
        }
    }
}


__global__ void run_density_kernels(int n, int* n_neighbors, int* neighbor_lst, int* neighbor_offset,
    GPU_T* x, GPU_T* y, GPU_T* density, GPU_T* shepard_correction_term, GPU_T smoothing_radius, GPU_T mass, GPU_T density_factor
) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < n) {

        int start_idx = neighbor_offset[i];
        int end_idx = neighbor_offset[i] + n_neighbors[i];

        for(int idx = start_idx; idx<end_idx; idx++) {
            int j = neighbor_lst[idx];
            GPU_T dst = std::sqrt(std::pow(x[i] - x[j],2) + std::pow(y[i] - y[j],2));

            density_kernel(i, j, dst, density, shepard_correction_term, smoothing_radius, mass, density_factor);
        }
        density_kernel(i, i, 0, density, shepard_correction_term, smoothing_radius, mass, density_factor);
    }
}

__global__ void apply_shepard_correction(int n, GPU_T* density, GPU_T* shepard_correction_term) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < n) {
        printf("%d %F\n", i, shepard_correction_term[i]);
        density[i] /= shepard_correction_term[i];        
    }
}

__global__ void run_force_kernels(int n, int* n_neighbors, int* neighbor_lst, int* neighbor_offset,
    GPU_T* x, GPU_T* y, GPU_T* vx, GPU_T* vy, GPU_T* force_x, GPU_T* force_y, GPU_T* density, 
    GPU_T smoothing_radius, GPU_T mass, GPU_T density_factor, GPU_T pressureMultiplier, GPU_T targetDensity, GPU_T dynamic_viscosity
) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < n) {
        int start_idx = neighbor_offset[i];
        int end_idx = neighbor_offset[i] + n_neighbors[i];
        for(int idx = start_idx; idx<end_idx; idx++) {
            int j = neighbor_lst[idx];

            GPU_T dst = std::sqrt(std::pow(x[i] - x[j],2) + std::pow(y[i] - y[j],2));

            pressure_force_kernel(i, j, dst, x, y, force_x, force_y, density, smoothing_radius, mass, density_factor, pressureMultiplier, targetDensity);
            viscous_force_kernel(i, j, x, y, vx, vy, force_x, force_y, density, smoothing_radius, mass, dynamic_viscosity, density_factor);
        }
    }
}


__global__ void time_integration_kernel(GPU_T* x, GPU_T* y, GPU_T* vx, GPU_T* vy, GPU_T* force_x, GPU_T* force_y, GPU_T* density, int n, GPU_T dt, int time_step) {
    int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    if (i < n) {
        if(time_step > 0) {
            vx[i] += dt * force_x[i] / density[i];
            vy[i] += dt * force_y[i] / density[i];
        } else {
            vx[i] += 0.5 * dt * force_x[i] / density[i];
            vy[i] += 0.5 * dt * force_y[i] / density[i];
        }
        
        x[i] += dt * vx[i];
        y[i] += dt * vy[i];
    }
}
