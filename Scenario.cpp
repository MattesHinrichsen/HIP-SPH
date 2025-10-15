#include "Utility/Timer.cpp"
#include "Utility/type.h"
#include "HashGrid/HashGrid.h"
#include "SPH/SPH_Simulator.h"
#include "Visualization/Visualizer.h"
#include <cmath>
#include <cstdint>
#include <format>

void Default_Scenario() {
    const int screenWidth = 1900;
    const int screenHeight = 900;
    const GPU_T max_x_coord = 19;
    const GPU_T max_y_coord = 9;

    int n = 8000;

    // Setup SPH Solver
        SPH_Simulator sph(n, max_x_coord, max_y_coord);
        
        // sph.load_from_file("./pos.bin");
        sph.Randomize_Points();
        // sph.Grid_Points();
        sph.store_to_file("./pos.bin");

        float pressureMultiplier = 50;
        float densityTarget = 45;
        sph.Set_Pressure_Multiplier(pressureMultiplier);
        sph.Set_Target_Density(densityTarget);
    
    // Setup visualization
        Visualizer vis(&sph,screenWidth, screenHeight, max_x_coord, max_y_coord);
        auto sim_time = std::chrono::high_resolution_clock::now();
        double sim_delta_time = 0;


    int iterations = 0;

    while (!vis.windowClosed()) {
    
        // Do simulation
        sim_time = std::chrono::high_resolution_clock::now();

        if(iterations % 100 == 0) {
            sph.z_order_sort();
        } 
        
        sph.Simulation_Step();
        sim_delta_time = std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - sim_time).count();


        // Do visualization
        vis.set_simulation_delta_time(sim_delta_time);
        vis.show_points(true);
        vis.draw_scene(120);
        vis.transfer_slider_changes();
        
        iterations++;
    }
}

void Benchmark_Scenario(int reorder_every = INT32_MAX) {
    const GPU_T max_x_coord = 16;
    const GPU_T max_y_coord = 9;

    int n = 100000;//4032;

    // Setup SPH Solver
        SPH_Simulator sph(n, max_x_coord, max_y_coord);
        sph.Randomize_Points();
        // sph.load_from_file("./pos.bin");
        // sph.Grid_Points();

        float pressureMultiplier = 50;
        float densityTarget = 45;
        sph.Set_Pressure_Multiplier(pressureMultiplier);
        sph.Set_Target_Density(densityTarget);

    {
        Timer t(std::format("ReorderEvery_{}", reorder_every));
        for(int iterations = 0; iterations < 20000; iterations++) {
            // Do simulation
            if(iterations % reorder_every == 0) sph.z_order_sort();
            sph.Simulation_Step();
        }
    }
    
    std::cout << "\n";
}