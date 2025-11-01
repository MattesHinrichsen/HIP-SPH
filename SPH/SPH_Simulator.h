#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <vector>
#include "raylib.h"

#include "../Utility/type.h"
#include "../Utility/hipHelper.cpp"
#include "../HashGrid/HashGrid.h"

class SPH_Simulator {

    int n;
    GPU_T mass;
    const GPU_T dt = 0.01;
    const GPU_T collisionDamping = 1.0;
    const GPU_T dynamic_viscosity = 2;

    GPU_T smoothing_radius;

    GPU_T density_factor;

    HashGrid hg;
    thrust::device_vector<GPU_T> vx;
    thrust::device_vector<GPU_T> vy;
    thrust::device_vector<GPU_T> density;
    thrust::device_vector<GPU_T> shepard_correction_term;
    thrust::device_vector<GPU_T> force_x;
    thrust::device_vector<GPU_T> force_y;

    std::vector<GPU_T> x_CPU;
    std::vector<GPU_T> y_CPU;

    std::vector<GPU_T> vx_CPU;
    std::vector<GPU_T> vy_CPU;

    GPU_T targetDensity; 
    GPU_T pressureMultiplier;

    int max_x;
    int max_y;

    int push_pull_mode;
    GPU_T push_pull_x;
    GPU_T push_pull_y;
    GPU_T push_pull_radius;

    bool performing_z_order_sort = false;

    int time_step = 0;

    GPU_T Convert_Density_GPU_To_Pressure(GPU_T density);

public:
    SPH_Simulator(int N_points, int max_x_coord, int max_y_coord);
    ~SPH_Simulator();

    void Simulation_Step();
    
    void Randomize_Points();
    void Grid_Points();
    void Time_Integration();
    void Calculate_Density();
    GPU_T Calculate_Density_At_Point(GPU_T x_pos, GPU_T y_pos);
    void Resolve_Collision();
    void Reset_Forces();
    void Set_Pull_Push_Mode(int mode, GPU_T x_pos, GPU_T y_pos);
    void Calculate_Pressure_and_Viscosity_Force();
    void Calculate_Gravity_Forces();
    void Calculate_Mouse_Pointer_Forces();
    
    void z_order_sort();
    void handle_visualization_interaction();
    bool is_performing_z_order_sort() { return performing_z_order_sort; }



    GPU_T Get_Smoothing_Radius();
    GPU_T Get_Scale(GPU_T scaling_factor = 0.5);
    int Get_Number_of_Points() { return n; }
    GPU_T Get_Target_Density() { return targetDensity; }
    GPU_T Get_Pressure_Multiplier() { return pressureMultiplier; }

    void Set_Smoothing_Radius(GPU_T radius);
    void Set_Target_Density(GPU_T target);
    void Set_Pressure_Multiplier(GPU_T multiplier);
    void Set_Push_Pull_Radius(GPU_T radius);

    void store_to_file(std::string path);
    void load_from_file(std::string path);

    std::vector<GPU_T>& get_x();
    std::vector<GPU_T>& get_y();
    std::vector<GPU_T>& get_vx();
    std::vector<GPU_T>& get_vy();

    // Debug
    void Print_Positions();
    void Print_Forces();
    void Print_Densities();
    void Print_Velocities();
};