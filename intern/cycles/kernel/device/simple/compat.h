#pragma once

#include <cstdint>
#include <math.h>
#include <cassert>
#include <climits>
#include <cstring>
#include <iostream>
#include <atomic>

#define __KERNEL_GPU__
#define __KERNEL_SIMPLE__
#define __KERNEL_64_BIT__
#define CCL_NAMESPACE_BEGIN
#define CCL_NAMESPACE_END 

#ifndef ATTR_FALLTHROUGH
#  define ATTR_FALLTHROUGH
#endif

#ifdef PRT_HIP_KERNEL
#  include "hip/hip_fp16.h"
#  include "hip/hip_runtime.h"
#endif

#define ccl_device inline
#define ccl_device_extern extern "C"
#define ccl_global
#define ccl_always_inline __attribute__((always_inline))
#define ccl_device_inline inline
#define ccl_noinline __attribute__((noinline))
#define ccl_inline_constant const constexpr
#define ccl_device_constant static constexpr
#define ccl_static_constexpr static constexpr
#define ccl_device_forceinline __attribute__((always_inline))
#define ccl_device_noinline ccl_device ccl_noinline
#define ccl_device_noinline_cpu ccl_device
#define ccl_device_inline_method ccl_device
#define ccl_restrict __restrict__
#define ccl_optional_struct_init
#define ccl_private
#define ccl_ray_data ccl_private
#define ccl_gpu_shared
#define ATTR_FALLTHROUGH __attribute__((fallthrough))
#define ccl_constant const
#define ccl_try_align(...) __attribute__((aligned(__VA_ARGS__)))
#define ccl_align(n) __attribute__((aligned(n)))
#define kernel_assert(cond)
#define ccl_may_alias


#define ccl_gpu_kernel_postfix
#define ccl_gpu_kernel(block_num_threads, thread_num_registers)
#define ccl_gpu_kernel_threads(block_num_threads)
#define ccl_gpu_shared
#define ccl_gpu_block_dim_x 1
#define ccl_gpu_thread_idx_x 0
#define ccl_gpu_warp_size 1
#define ccl_gpu_block_idx_x global_idx



#define ccl_gpu_ballot(predicate) (predicate ? 1 : 0)
#define ccl_gpu_thread_mask(thread_warp) \
    ((thread_warp) >= 1 ? 1 : 0)  // Máscara para 1 hilo
#define ccl_gpu_kernel_call(x) x



#if defined(PRT_SYCL_KERNEL) || defined(PRT_EMBREE_SYCL_KERNEL) 
#  define ccl_gpu_syncthreads() sycl::ext::oneapi::this_work_item::get_nd_item<1>().barrier()
#  define ccl_gpu_thread_idx_x (sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_local_id(0))
#  define ccl_gpu_block_dim_x (sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_local_range(0))
#  define ccl_gpu_block_idx_x (sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_group(0))
#  define ccl_gpu_grid_dim_x (sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_group_range(0))
#  define ccl_gpu_warp_size (sycl::ext::oneapi::this_work_item::get_sub_group().get_local_range()[0])
#  define ccl_gpu_global_id_x() (sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_id(0))
#  define ccl_gpu_global_size_x() (sycl::ext::oneapi::this_work_item::get_nd_item<1>().get_global_range(0))
#  define ccl_gpu_ballot(predicate) (sycl::ext::oneapi::group_ballot(sycl::ext::oneapi::this_work_item::get_sub_group(), predicate).count())
#  define ccl_gpu_thread_mask(thread_warp) uint(0xFFFFFFFF >> (ccl_gpu_warp_size - thread_warp))
#else 
#define ccl_gpu_syncthreads void
#define ccl_gpu_global_id_x() (ccl_gpu_block_idx_x * ccl_gpu_block_dim_x + \
  ccl_gpu_thread_idx_x)
#endif





#define kernel_data kernel_globals.__data
#define kernel_integrator_state kernel_globals.integrator_state
#define ccl_device_inline inline
#define ccl_device_inline_method ccl_device
#define ccl_private
#define ccl_try_align(...) __attribute__((aligned(__VA_ARGS__)))
#define ccl_device_template_spec template<> ccl_device_inline
#define ccl_device_forceinline static inline
#define ccl_global
#define ccl_align(n) __attribute__((aligned(n)))
#define ccl_restrict __restrict__
#define ccl_static_constexpr static constexpr
#define ccl_device_noinline __attribute__((noinline))
#define ccl_inline_constant const constexpr
#define ccl_constant const
#define kernel_assert(cond)
#define ccl_ray_data ccl_private
#define ccl_gpu_kernel_within_bounds(i, n) ((i) < (n))
#define ccl_optional_struct_init
#define ccl_device_noinline_cpu ccl_device




#if defined(PRT_OPTIX_KERNEL)
  #define OPTIX_DONT_INCLUDE_CUDA
  #include <optix_device.h>
  #define ccl_device \
    static __device__ \
        __forceinline__  // Function calls are bad for OptiX performance, so inline everything
  #define ccl_device_extern extern "C" __device__
  #define ccl_device_inline ccl_device
  #define ccl_device_forceinline ccl_device
  #define ccl_device_inline_method __device__ __forceinline__
  #define ccl_device_noinline static __device__ __noinline__
  #define ccl_device_noinline_cpu ccl_device
  #define ccl_global
  #define ccl_inline_constant static constexpr __constant__
  #define ccl_device_constant __constant__ __device__
  #define ccl_static_constexpr static constexpr
  #define ccl_constant const
  #define ccl_gpu_shared __shared__
  #define ccl_private
  #define ccl_ray_data ccl_private
  #define ccl_may_alias
  #define ccl_restrict __restrict__
  #define ccl_align(n) __align__(n)
  #define ccl_gpu_syncthreads()
#elif defined(PRT_CUDA_KERNEL)
  #define ccl_device __device__ __inline__
  #define ccl_device_extern extern "C" __device__
  #if __CUDA_ARCH__ < 500
  #  define ccl_device_inline __device__ __forceinline__
  #  define ccl_device_forceinline __device__ __forceinline__
  #else
  #  define ccl_device_inline __device__ __inline__
  #  define ccl_device_forceinline __device__ __forceinline__
  #endif
  #define ccl_device_noinline __device__ __noinline__
  #define ccl_device_noinline_cpu ccl_device
  #define ccl_device_inline_method ccl_device
  #define ccl_global
  #define ccl_inline_constant static constexpr __constant__
  #define ccl_device_constant __constant__ __device__
  #define ccl_static_constexpr static constexpr
  #define ccl_constant const
  #define ccl_gpu_shared __shared__
  #define ccl_private
  #define ccl_ray_data ccl_private
  #define ccl_may_alias
  #define ccl_restrict __restrict__
  #define ccl_align(n) __align__(n)
  #define ccl_optional_struct_init
  #define ccl_gpu_syncthreads() __syncthreads()
#elif defined(PRT_HIP_KERNEL)
  #define ccl_device __device__ __inline__
  #define ccl_device_extern extern "C" __device__
  #define ccl_device_inline __device__ __inline__
  #define ccl_device_forceinline __device__ __forceinline__
  #define ccl_device_noinline __device__ __noinline__
  #define ccl_device_noinline_cpu ccl_device
  #define ccl_device_inline_method ccl_device
  #define ccl_global
  #define ccl_inline_constant inline constexpr
  #define ccl_device_constant __constant__ __device__
  #define ccl_static_constexpr static constexpr
  #define ccl_constant const
  #define ccl_private
  #define ccl_ray_data ccl_private
  #define ccl_may_alias
  #define ccl_restrict __restrict__
  #define ccl_align(n) __align__(n)
  //#define ccl_gpu_syncthreads() __syncthreads()
  //#define ccl_gpu_warp_size (warpSize)
  //#define ccl_gpu_thread_mask(thread_warp) uint64_t((1ull << thread_warp) - 1)
  //#define ccl_gpu_syncthreads() __syncthreads()
  //#define ccl_gpu_ballot(predicate) __ballot(predicate)
#else
#define __device__
#endif

#if defined(PRT_EMBREE_SYCL_KERNEL) || defined(PRT_SYCL_KERNEL)

#define fabsf(x) sycl::fabs((x))
#define copysignf(x, y) sycl::copysign((x), (y))
#define asinf(x) sycl::asin((x))
#define acosf(x) sycl::acos((x))
#define atanf(x) sycl::atan((x))
#define floorf(x) sycl::floor((x))
#define ceilf(x) sycl::ceil((x))
#define sinhf(x) sycl::sinh((x))
#define coshf(x) sycl::cosh((x))
#define tanhf(x) sycl::tanh((x))
#define hypotf(x, y) sycl::hypot((x), (y))
#define atan2f(x, y) sycl::atan2((x), (y))
#define fmaxf(x, y) sycl::fmax((x), (y))
#define fminf(x, y) sycl::fmin((x), (y))
#define fmodf(x, y) sycl::fmod((x), (y))
#define lgammaf(x) sycl::lgamma((x))

#define cosf(x) sycl::native::cos(((float)(x)))
#define sinf(x) sycl::native::sin(((float)(x)))
#define powf(x, y) sycl::native::powr(((float)(x)), ((float)(y)))
#define tanf(x) sycl::native::tan(((float)(x)))
#define logf(x) sycl::native::log(((float)(x)))
#define expf(x) sycl::native::exp(((float)(x)))
#define sqrtf(x) sycl::native::sqrt(((float)(x)))

#define ccl_device
#define ccl_device_extern extern "C"
#define ccl_global
#define ccl_always_inline __attribute__((always_inline))
#define ccl_device_inline inline
#define ccl_noinline __attribute__((noinline))
#define ccl_inline_constant const constexpr
#define ccl_device_constant static constexpr
#define ccl_static_constexpr static constexpr
#define ccl_device_forceinline __attribute__((always_inline))
#define ccl_device_noinline 
#define ccl_device_noinline_cpu 
#define ccl_device_inline_method ccl_device
#define ccl_restrict __restrict__
#define ccl_optional_struct_init
#define ccl_private
#define ccl_ray_data ccl_private
#define ccl_gpu_shared
#define ATTR_FALLTHROUGH __attribute__((fallthrough))
#define ccl_constant const
#define ccl_try_align(...) __attribute__((aligned(__VA_ARGS__)))
#define ccl_align(n) __attribute__((aligned(n)))
#define kernel_assert(cond)
#define ccl_may_alias

using uchar = unsigned char;
using sycl::half;

/* math functions */
ccl_device_forceinline float __uint_as_float(unsigned int x)
{
  return sycl::bit_cast<float>(x);
}
ccl_device_forceinline unsigned int __float_as_uint(const float x)
{
  return sycl::bit_cast<unsigned int>(x);
}
ccl_device_forceinline float __int_as_float(const int x)
{
  return sycl::bit_cast<float>(x);
}
ccl_device_forceinline int __float_as_int(const float x)
{
  return sycl::bit_cast<int>(x);
}

#endif

#define ccl_gpu_kernel_signature(name, ...) prt_kernel void simple_##name(__VA_ARGS__)
//#define ccl_gpu_kernel_signature(name, ...) PRT_KERNEL(simple_##name, __VA_ARGS__)

#define ccl_gpu_kernel_lambda(func, ...) \
  struct KernelLambda \
  { \
    KernelLambda(const prt::KernelGlobals *_kg) : kg(_kg) {} \
    ccl_private const prt::KernelGlobals *kg; \
    __VA_ARGS__; \
    int operator()(const int state) const { return (func); } \
  } ccl_gpu_kernel_lambda_pass((prt::KernelGlobals *)__prt_kg)



struct CPUTexture2D {
  const void *pixels;    /* pointer a la imagen en memoria lineal   */
  int         width;
  int         height;
  int         channels;  /* 1, 3, 4 …                                */
  int         data_type;
  int         wrap_type;
  /* flags de interpolación, extensión, etc. si los necesitas        */
};

struct CPUTexture3D {
  const void *voxels;
  int         width, height, depth;
  int         channels;
};


using ccl_gpu_tex_object_2D = const CPUTexture2D *;
using ccl_gpu_tex_object_3D = const CPUTexture3D *;

ccl_device_inline int clampi(int x, int lo, int hi)
{
  return (x < lo) ? lo : (x > hi ? hi : x);
}

#if defined(PRT_CPU_KERNEL) || defined(PRT_EMBREE_CPU_KERNEL)

// TODO esto es solo para cpu, mover a su sitio correspondiente.
// uint32/int: fetch_add/sub devuelven el valor viejo
static inline uint32_t _afaa_u32(unsigned int* p, uint32_t v) {
    return __atomic_fetch_add(p, v, __ATOMIC_RELAXED);
}
static inline uint32_t _afaa_u32(int* p, uint32_t v) {
    return (uint32_t)__atomic_fetch_add(p, (int)v, __ATOMIC_RELAXED);
}
static inline uint32_t _afas_u32(unsigned int* p, uint32_t v) {
    return __atomic_fetch_sub(p, v, __ATOMIC_RELAXED);
}
static inline uint32_t _afas_u32(int* p, uint32_t v) {
    return (uint32_t)__atomic_fetch_sub(p, (int)v, __ATOMIC_RELAXED);
}

// float: add-and-fetch devuelve el nuevo valor
static inline float _aafe_f32(float* p, float v) {
    float old;
    __atomic_load(p, &old, __ATOMIC_RELAXED);
    for (;;) {
        float desired = old + v;
        if (__atomic_compare_exchange(p, &old, &desired, true,
                                      __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            return desired; // nuevo
        // 'old' queda actualizado al valor actual si falló, reintentamos
    }
}

// float: CAS que devuelve el valor previo (útil para leer el ID anterior)
static inline float _acas_f32(float* p, float expected, float desired) {
    float exp = expected;
    (void)__atomic_compare_exchange(p, &exp, &desired, false,
                                    __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    return exp; // valor que había antes (éxito o no)
}

#define atomic_fetch_and_add_uint32(ptr, val) _afaa_u32((ptr), (uint32_t)(val))
#define atomic_fetch_and_sub_uint32(ptr, val) _afas_u32((ptr), (uint32_t)(val))
#define atomic_add_and_fetch_float(ptr, val)  _aafe_f32((ptr), (float)(val))
#define atomic_compare_and_swap_float(ptr, oldval, newval) _acas_f32((ptr), (float)(oldval), (float)(newval))

#elif defined(PRT_OPTIX_KERNEL) || defined(PRT_CUDA_KERNEL)

typedef unsigned short half;

ccl_device_forceinline half __float2half(const float f)
{
  half val;
  asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(val) : "f"(f));
  return val;
}

ccl_device_forceinline float __half2float(const half h)
{
  float val;
  asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(h));
  return val;
}
#endif

