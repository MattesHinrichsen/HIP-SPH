cmake_minimum_required(VERSION 3.30)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# LLVM target name: https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
set(GPU_Architecture gfx1101)

# Compiler Flags
set(CompilationMode "-O3;-march=native")
# set(CompilationMode "-g;-Og;-march=native")


# Variable for the project name
set(ProjName "Simulation")

# Create project
project(${ProjName} VERSION 1.0 LANGUAGES CXX HIP)


add_executable(${ProjName} main.cpp)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)

# Add Libraries
find_library(RAYLIB raylib)
target_link_libraries(${ProjName} PUBLIC ${RAYLIB})
find_library(IMGUI imgui)
target_link_libraries(${ProjName} PUBLIC ${IMGUI})


# Add compile options
target_compile_options(${ProjName} PUBLIC ${CompilationMode})
add_definitions(-w)


set(SPH_Files "SPH/SPH_Simulator.cpp")
add_library(SPH SHARED ${SPH_Files})
target_compile_options(SPH PUBLIC ${CompilationMode})
set_property(TARGET SPH PROPERTY HIP_ARCHITECTURES ${GPU_Architecture})
set_source_files_properties(${SPH_Files} PROPERTIES LANGUAGE HIP)
find_library(ROCRAND rocrand PATHS "/opt/rocm/lib/")
target_link_libraries(SPH PUBLIC ${ROCRAND})

set(HashGrid_Files "HashGrid/HashGrid.cpp" "HashGrid/HashGrid_Kernels.cpp")
add_library(HashGrid SHARED ${HashGrid_Files})
target_compile_options(HashGrid PUBLIC ${CompilationMode})
set_property(TARGET HashGrid PROPERTY HIP_ARCHITECTURES ${GPU_Architecture})
set_source_files_properties(${HashGrid_Files} PROPERTIES LANGUAGE HIP)

set(Visualization_Files "Visualization/Visualizer.cpp")
add_library(Visualization SHARED ${Visualization_Files})
target_compile_options(Visualization PUBLIC ${CompilationMode})
set_property(TARGET Visualization PROPERTY HIP_ARCHITECTURES ${GPU_Architecture})
set_source_files_properties(${Visualization_Files} PROPERTIES LANGUAGE HIP)
target_compile_definitions(Visualization PRIVATE RL_FLOAT3_TYPE)


target_link_libraries(${ProjName} PUBLIC SPH)
target_link_libraries(${ProjName} PUBLIC HashGrid)
target_link_libraries(${ProjName} PUBLIC Visualization)



# Add make run command
add_custom_target(run
    COMMAND ${ProjName}
    DEPENDS ${ProjName}
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)


