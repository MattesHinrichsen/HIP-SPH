
# Dependency installation on Linux (Fedora 42, Wayland)

### AMD HIP + ROCM
1. Download the installer from <https://www.amd.com/en/support/linux-drivers>.
2. Install it via `sudo dnf install amdgpu-install...`.
3. The next two commands might be optional: Run `sudo sed -i -e "s#\$amdgpudistro#9.3#g" /etc/yum.repos.d/amdgpu*.repo` and `sudo dnf config-manager --enable rocm`.
4. Finally install HIP via `amdgpu-install --no-dkms --usecase=hip,hiplibsdk,rocmdevtools`. The no-dkms option might be critical.

Source: <https://blog.onlyesterday.net/post/tech/2024/05/30_radeon_fedora40/#gsc.tab=0>


### Raylib
Check <https://github.com/raysan5/raylib/wiki/Working-on-GNU-Linux>

1. Run `git clone --depth 1 https://github.com/raysan5/raylib.git raylib`
2. `cd raylib/src/`
3. `make PLATFORM=PLATFORM_DESKTOP`
4. Copy libraylib.a to /usr/local/lib using `sudo cp libraylib.a /usr/local/lib`.
5. Copy raylib.h to /usr/local/include using `sudo cp *.h /usr/local/include`.


### IMGUI and rlIMGUI
1. Create a new temporary directory.
2. Copy all .cpp files of IMGUI and rlIMGUI into temporary folder.
3. Create an include directory.
4. Copy all .h files of IMGUI and rlIMGUI into include folder.
5. In temporary folder run commands 
   ```
   g++ -c *.cpp -I"./include" 
   ar rvs libimgui.a *.o 
   ```
6. Copy the created file "libimgui.a" to the directory /usr/local/lib using `sudo cp libimgui.a /usr/local/lib`
7. Copy the contents of the include folder to /usr/local/include. Also copy `extras` folder. Using `sudo cp -r ./include/* /usr/local/include`