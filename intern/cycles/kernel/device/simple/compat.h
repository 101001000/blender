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
#define CCL_NAMESPACE_BEGIN
#define CCL_NAMESPACE_END 

#ifndef ATTR_FALLTHROUGH
#  define ATTR_FALLTHROUGH
#endif

#ifdef HIP_KERNEL
#  include "hip/hip_fp16.h"
#  include "hip/hip_runtime.h"
#endif

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

#ifdef SYCL_KERNEL
#define ccl_gpu_syncthreads() sycl::ext::oneapi::this_work_item::get_nd_item<1>().barrier()
#else 
#define ccl_gpu_syncthreads void
#endif

#define ccl_gpu_ballot(predicate) (predicate ? 1 : 0)

#define ccl_gpu_kernel_call(x) x

#define ccl_gpu_thread_mask(thread_warp) \
    ((thread_warp) >= 1 ? 1 : 0)  // Máscara para 1 hilo

#define ccl_gpu_global_id_x() (ccl_gpu_block_idx_x * ccl_gpu_block_dim_x + \
  ccl_gpu_thread_idx_x)
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



#ifdef OPTIX_KERNEL 
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
  #define ccl_inline_constant static __constant__
  #define ccl_device_constant __constant__ __device__
  #define ccl_static_constexpr static constexpr
  #define ccl_constant const
  #define ccl_gpu_shared __shared__
  #define ccl_private
  #define ccl_ray_data ccl_private
  #define ccl_may_alias
  #define ccl_restrict __restrict__
  #define ccl_align(n) __align__(n)
#elif defined(HIP_KERNEL)
  #define ccl_device __device__ __inline__
  #define ccl_device_extern extern "C" __device__
  #define ccl_device_inline __device__ __inline__
  #define ccl_device_forceinline __device__ __forceinline__
  #define ccl_device_noinline __device__ __noinline__
  #define ccl_device_noinline_cpu ccl_device
  #define ccl_device_inline_method ccl_device
  #define ccl_global
  #define ccl_inline_constant __constant__
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
#else
#define __device__
#endif

#define ccl_gpu_kernel_signature(name, ...) PRT_KERNEL(simple_##name, __VA_ARGS__)

#define ccl_gpu_kernel_lambda(func, ...) \
  struct KernelLambda { \
    __VA_ARGS__; \
    int operator()(const int state) \
    { \
      return (func); \
    } \
  } ccl_gpu_kernel_lambda_pass


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

#if defined(CPU_KERNEL) || defined(EMBREE_CPU_KERNEL)

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

#elif defined(OPTIX_KERNEL)
#    define atomic_add_and_fetch_float(p, x) (atomicAdd((float *)(p), (float)(x)) + (float)(x))
#    define atomic_fetch_and_add_uint32(p, x) atomicAdd((unsigned int *)(p), (unsigned int)(x))
#    define atomic_fetch_and_sub_uint32(p, x) atomicSub((unsigned int *)(p), (unsigned int)(x))
ccl_device_inline float atomic_compare_and_swap_float(volatile float *dest,
                                                      const float old_val,
                                                      const float new_val)
{
  union {
    unsigned int int_value;
    float float_value;
  } new_value, prev_value, result;
  prev_value.float_value = old_val;
  new_value.float_value = new_val;
  result.int_value = atomicCAS((unsigned int *)dest, prev_value.int_value, new_value.int_value);
  return result.float_value;
}

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

