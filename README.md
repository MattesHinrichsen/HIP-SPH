# HIP-SPH
Two-dimensional Smoothed Particle Hydrodynamics solver written using AMD HIP with Hashgrid-based neighborhood search.

![Demo](Utility/SPH_Demo2.gif)

# Setup
1. Install dependencies: See file `InstallHints.md` for guides.
   - AMD HIP + ROCM: CMake assumes this to be located at `/opt/rocm`.
   - Raylib
   - IMGUI
   - rlIMGUI
2. Set the correct GPU deviceID in `Utility/hipHelper.cpp`. The command `rocminfo` might give you more info on the available devices. The used device is printed when the simulation is started.
3. In main directory run 
   ```
   cd build
   cmake ..
   make run -j4
   ```
