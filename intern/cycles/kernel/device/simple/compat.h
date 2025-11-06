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

#define ccl_gpu_kernel_signature(name, ...) PRT_KERNEL(simple_##name, __VA_ARGS__)
#define ccl_gpu_kernel_postfix
#define ccl_gpu_kernel(block_num_threads, thread_num_registers)
#define ccl_gpu_kernel_threads(block_num_threads) 
#define ccl_device
#define ccl_gpu_shared
#define ccl_gpu_block_dim_x 1
#define ccl_gpu_thread_idx_x 0
#define ccl_gpu_warp_size 1
#define ccl_gpu_block_idx_x global_idx
#define ccl_gpu_syncthreads void
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
#define __device__
#define ccl_gpu_global_id_x() global_idx

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



