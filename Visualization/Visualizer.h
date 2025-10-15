#pragma once
#include "../SPH/SPH_Simulator.h"
#include "raylib.h"
#include "rlImGui.h"
#include "imgui.h"
#include <chrono>
#include <future>
#include <vector>
#include "../Utility/type.h"

class Visualizer {

    SPH_Simulator* sph;

    int screenWidth = 900;
    int screenHeight = 900;
    GPU_T max_x_coord;
    GPU_T max_y_coord;
    int fps = 60;

    GPU_T scaling_factor = 0.5;
    GPU_T scale = 1;
    GPU_T smoothing_radius;
    GPU_T densityTarget;
    GPU_T pressureMultiplier;

    GPU_T push_pull_radius;

    std::chrono::time_point<std::chrono::high_resolution_clock> vis_time = std::chrono::high_resolution_clock::now();
    std::chrono::time_point<std::chrono::high_resolution_clock> time_last_draw = std::chrono::high_resolution_clock::now(); 
    double vis_delta_time = 0;
    double sim_delta_time = -1;

    std::future<void> draw_future;

    bool draw_points = true;
    bool draw_velocity = true;
    bool draw_requested = false;

    bool closeWindow = false;

    Texture2D circle_texture;

    std::vector<Color> color_gradient;
    GPU_T max_vel = 1;
    GPU_T min_vel = 0.1;

    Camera2D camera;

    void draw_radius(int push_pull_mode);
    void drawing_call();

    // Main drawing loop running on second thread 
    void main();

    void handle_zoom_and_move();

    void set_color_gradient();

public:
    Visualizer(SPH_Simulator* SPH, int sWidth, int sHeight, GPU_T max_x, GPU_T max_y);
    
    ~Visualizer() { if(draw_future.valid()) draw_future.wait(); }

    void show_points(bool dpos) { draw_points = dpos; }
    void show_velocity(bool dvel) { draw_velocity = dvel; }

    void set_simulation_delta_time(GPU_T dt) { sim_delta_time = dt; }

    // Tell the visualizer to start the drawing sequence
    void draw_scene(int targetFPS);

    bool windowClosed() { return closeWindow; }

    void transfer_slider_changes();

    // Visualization coordinates to simulation coordinates
    GPU_T x_vts(GPU_T val) { return val / screenWidth * max_x_coord; }
    GPU_T y_vts(GPU_T val) { return (screenHeight - val) / screenHeight * max_y_coord; }
    Vector2 vts(GPU_T x, GPU_T y) {
        auto world_pos = GetScreenToWorld2D(
            {
                x,
                y
            },
            camera
        );

        return {x_vts(world_pos.x), y_vts(world_pos.y)};
    }

    // Simulation coordinates to visualization coordinates
    GPU_T x_stv(GPU_T val) { return val / max_x_coord * screenWidth; }
    GPU_T y_stv(GPU_T val) { return screenHeight - (val / max_y_coord * screenHeight); }

};