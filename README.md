# HIP-SPH
Two-dimensional Smooted Particle Hydrodynamics solver written using AMD HIP with hashgrid-based neighborhood search.

![Demo](Utility/SPH_Demo.gif)

# Setup
1. Install dependencies: See file `InstallHints.md` for guides.
   - AMD HIP + ROCM: CMake assumes this to be located at `/opt/rocm`.
   - Raylib
   - IMGUI
   - rlIMGUI
2. In main directory run 
   ```
   cd Build
   cmake ..
   make run -j4
   ```
