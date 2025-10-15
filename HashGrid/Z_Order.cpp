#include <hip/hip_runtime.h> 
#include <sys/types.h>
#include "../Utility/type.h"

inline __device__ u_int32_t float_to_int(GPU_T val, GPU_T min_val, GPU_T max_val) {
    GPU_T clamped = (val-min_val) / (max_val - min_val);
   return static_cast<u_int32_t>(clamped * ((1u << 16) - 1));
}

inline __device__ u_int32_t part1by1(u_int32_t n) {
   n &= 0x0000FFFF;
   n = (n | (n << 8)) & 0x00FF00FF;
   n = (n | (n << 4)) & 0x0F0F0F0F;
   n = (n | (n << 2)) & 0x33333333;
   n = (n | (n << 1)) & 0x55555555;
   return n;
}

inline __device__ u_int32_t morton_code(u_int32_t x, u_int32_t y) {
   return (part1by1(x) << 1) | part1by1(y);
}