#include "Visualizer.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <future>
#include <imgui.h>
#include <iostream>
#include <raylib.h>
#include "raymath.h"
#include <vector>

using namespace std::chrono_literals;



Visualizer::Visualizer(SPH_Simulator* SPH, int sWidth, int sHeight, GPU_T max_x, GPU_T max_y) 
        : screenWidth(sWidth), screenHeight(sHeight), sph(SPH), max_x_coord(max_x), max_y_coord(max_y) {

    assert(max_y/sHeight == max_x/sWidth);
    scale = (GPU_T)std::max(screenWidth, screenHeight) / (300*std::sqrt(sph->Get_Number_of_Points())) * scaling_factor;
    smoothing_radius = sph->Get_Smoothing_Radius();
    densityTarget = sph->Get_Target_Density();
    pressureMultiplier = sph->Get_Pressure_Multiplier();
    push_pull_radius = smoothing_radius;

    color_gradient = std::vector<Color>(1024);
    set_color_gradient();
    
    // Call drawing main
    draw_future = std::async(std::launch::async, [&](){
        main();
    });

}

void Visualizer::draw_radius(int push_pull_mode) {
    auto mouse_pos = GetScreenToWorld2D({(float)GetMouseX(), (float)GetMouseY()}, camera);
    int mouse_x = mouse_pos.x;
    int mouse_y = mouse_pos.y;
    if (!push_pull_mode) DrawCircleLines(mouse_x, mouse_y, x_stv(smoothing_radius), RED);
    else DrawCircleLines(mouse_x, mouse_y, x_stv(push_pull_radius), BLUE);
}


void Visualizer::handle_zoom_and_move() {
    // Translate based on mouse right click
    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT) and IsKeyDown(KEY_LEFT_CONTROL))
    {
        Vector2 delta = GetMouseDelta();
        delta = Vector2Scale(delta, -1.0f/camera.zoom);
        camera.target = Vector2Add(camera.target, delta);
    }

    // Zoom based on mouse wheel
    float wheel = GetMouseWheelMove();
    if (wheel != 0 and IsKeyDown(KEY_LEFT_CONTROL))
    {
        // Get the world point that is under the mouse
        Vector2 mouseWorldPos = GetScreenToWorld2D(GetMousePosition(), camera);

        // Set the offset to where the mouse is
        camera.offset = GetMousePosition();

        // Set the target to match, so that the camera maps the world space point 
        // under the cursor to the screen space point under the cursor at any zoom
        camera.target = mouseWorldPos;

        // Zoom increment
        // Uses log scaling to provide consistent zoom speed
        float scale = 0.2f*wheel;
        camera.zoom = Clamp(expf(logf(camera.zoom)+scale), 0.125f, 64.0f);
    }
}




void Visualizer::drawing_call() {
    vis_time = std::chrono::high_resolution_clock::now();
    handle_zoom_and_move();

    BeginDrawing();

        // Setup
        ClearBackground(RAYWHITE);

        // DrawFPS(2, 2);

        rlImGuiBegin();
        ImGui::Begin("Developer Settings");
        ImGui::Text("Application Framerate: %.1f FPS (max %.1f FPS)", double(ImGui::GetIO().Framerate), std::min(1000000./sim_delta_time, 1000000./vis_delta_time));
        ImGui::Text("Visualization Time: %.3f ms/frame (%.1f FPS)", vis_delta_time/1000., 1000000./vis_delta_time);
        ImGui::Text("Simulation Time: %.3f ms/iteration (%.1f FPS)", sim_delta_time/1000., 1000000./sim_delta_time);
        sim_delta_time = -1;


        BeginMode2D(camera);
            // Scaling Factor
            bool sliderChanged = ImGui::SliderFloat("Scaling Factor", &scaling_factor, 0, 2);
            if (sliderChanged){
                int n = sph->Get_Number_of_Points();
                scale = (GPU_T)std::max(screenWidth, screenHeight) / (300*std::sqrt(n)) * scaling_factor;
            }


            // Draw points
            int n = sph->Get_Number_of_Points();
            if(draw_points) {

                // If a sort is being performed dont copy
                while(sph->is_performing_z_order_sort()) {
                    std::this_thread::sleep_for(10us);
                }

                auto x = sph->get_x();
                auto y = sph->get_y();
                GPU_T* vx = nullptr;
                GPU_T* vy = nullptr;
                if (draw_velocity) {
                    vx = sph->get_vx().data();
                    vy = sph->get_vy().data();
                }

                for(int i = 0; i<n; i++) {
                    float x_pos = x_stv(x[i])  - 300*scale/2.;
                    float y_pos = y_stv(y[i]) - 300*scale/2.;

                    Color c;
                    if (draw_velocity) {
                        GPU_T v = std::sqrt(vx[i] * vx[i] + vy[i] * vy[i]);
                        int idx = (std::clamp(v, min_vel, max_vel) - min_vel) / (max_vel-min_vel) * 1023;
                        c = color_gradient[idx];
                    } else {
                        c = BLACK;
                    }


                    DrawTextureEx(circle_texture, {x_pos, y_pos}, 0.f, scale, c);
                }
            }

            // Draw Velocity
        

            // Find Density at mouse cursor
            auto mouse_pos = vts(GetMouseX(), GetMouseY());
            ImGui::Text("Pressure at cursor: %f", sph->Calculate_Density_At_Point(mouse_pos.x, mouse_pos.y)); 

            // Setting smoothing radius
            smoothing_radius = sph->Get_Smoothing_Radius();
            sliderChanged = ImGui::SliderFloat("Smoothing Radius", &smoothing_radius, 1e-5, 2);

            // Set target density
            densityTarget = sph->Get_Target_Density();
            sliderChanged = ImGui::SliderFloat("Target Density", &densityTarget, 0, 500);
        
            // Set pressure multiplier
            pressureMultiplier = sph->Get_Pressure_Multiplier();
            sliderChanged = ImGui::SliderFloat("Pressure Multiplier", &pressureMultiplier, 0, 100);
        
            // Perform a simulation step
            bool button_pressed = ImGui::Button("Simulation Step");
            if(button_pressed) {
                sph->Simulation_Step();
            }

            // Pull or push force
            int push_pull_mode = 0;
            if(IsMouseButtonDown(MOUSE_LEFT_BUTTON) and not IsKeyDown(KEY_LEFT_CONTROL)) {


                auto mouse_pos = vts(GetMouseX(), GetMouseY());
                sph->Set_Pull_Push_Mode(1, mouse_pos.x, mouse_pos.y);
                push_pull_mode = 1;
                push_pull_radius += 5 * x_vts(GetMouseWheelMove());
            } 
            if(IsMouseButtonDown(MOUSE_RIGHT_BUTTON) and not IsKeyDown(KEY_LEFT_CONTROL)) {
                auto mouse_pos = vts(GetMouseX(), GetMouseY());
                sph->Set_Pull_Push_Mode(2, mouse_pos.x, mouse_pos.y);
                push_pull_mode = 2;
                push_pull_radius += 5 * x_vts(GetMouseWheelMove());
            } 
            

            // Draw the smoothing radius
            draw_radius(push_pull_mode);
        EndMode2D();

        ImGui::End();
        rlImGuiEnd();
    EndDrawing();

    vis_delta_time = std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - vis_time).count();
}


void Visualizer::draw_scene(int targetFPS) {
    fps = targetFPS; 
    draw_requested = true;
}

void Visualizer::transfer_slider_changes() {
    sph->Set_Pressure_Multiplier(pressureMultiplier);
    sph->Set_Smoothing_Radius(smoothing_radius);
    sph->Set_Target_Density(densityTarget);
    sph->Set_Push_Pull_Radius(push_pull_radius);
}


void Visualizer::main() {

    // Setup
    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    SetConfigFlags(FLAG_MSAA_4X_HINT); 
    SetTraceLogLevel(LOG_ERROR); 
    InitWindow(screenWidth, screenHeight, "SPH");
    rlImGuiSetup(true);

    circle_texture = LoadTexture("../Utility/CircleWhite.png");
    camera = { 0 };
    camera.zoom = 1.0f;


    // Main Loop
    while(!WindowShouldClose()) {
        GPU_T dt = std::chrono::duration<double, std::micro>(std::chrono::high_resolution_clock::now() - time_last_draw).count();
        if (draw_requested and 1000000/dt < fps) {
            time_last_draw = std::chrono::high_resolution_clock::now();
            draw_requested = false;
            drawing_call();
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    closeWindow = true;

}

void Visualizer::set_color_gradient() {
    int idx = 0;
    for(int i = 0; i<256; i++) {
        color_gradient[idx] = Color(255,i,0,255);
        idx++;
    }
    for(int i = 0; i<256; i++) {
        color_gradient[idx] = Color(255-i,255,0,255);
        idx++;
    }
    for(int i = 0; i<256; i++) {
        color_gradient[idx] = Color(0,255,0,i);
        idx++;
    }
    for(int i = 0; i<256; i++) {
        color_gradient[idx] = Color(0,255-i,255,255);
        idx++;
    }

    std::reverse(color_gradient.begin(), color_gradient.end());
}
