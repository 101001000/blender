// TODO: Limpiar esto
#ifdef OPTIX_KERNEL
  #include "kernel/device/optix/compat.h"
  #define ccl_gpu_block_dim_x 1
  #define ccl_gpu_thread_idx_x 0
  #define ccl_gpu_warp_size 1
  #define ccl_gpu_block_idx_x 0
  #define ccl_gpu_ballot 1
  #define ccl_gpu_thread_mask 1
  #define ccl_gpu_syncthreads void
  #define ccl_gpu_block_idx_x global_idx
  #define ccl_gpu_ballot(predicate) (predicate ? 1 : 0)
  #define ccl_gpu_kernel_call(x) x
  #define ccl_gpu_thread_mask(thread_warp) \
    ((thread_warp) >= 1 ? 1 : 0)  // Máscara para 1 hilo
  #define ccl_gpu_global_id_x() (ccl_gpu_block_idx_x * ccl_gpu_block_dim_x + \
    ccl_gpu_thread_idx_x)
  #define ccl_gpu_shared

  //__constant__ int warp_offset[128 * 10 + 1];
  #undef __KERNEL_OPTIX__
  #define __KERNEL_GPU__
  #define __KERNEL_SIMPLE__
#endif

#if defined(CPU_KERNEL) || defined(EMBREE_CPU_KERNEL) || defined(SYCL_KERNEL) 
  #include "kernel/device/simple/compat.h"
  #include "util/half.h"
  #include "util/types.h"

  
  template<typename T>
  ccl_device_forceinline T ccl_gpu_tex_object_read_2D(const ccl_gpu_tex_object_2D texobj,
                                                      const float fx, const float fy)
  {

    const CPUTexture2D* tex = reinterpret_cast<const CPUTexture2D*>(texobj);
    const float u = fx * tex->width  - 0.5f;
    const float v = fy * tex->height - 0.5f;
    const int channels = tex->channels;
    const int ix = clampi(int(std::floor(u + 0.5f)), 0, tex->width  - 1);
    const int iy = clampi(int(std::floor(v + 0.5f)), 0, tex->height - 1);
    const size_t idx = ((size_t)iy * tex->width + ix);
    const unsigned char *ptr = reinterpret_cast<const unsigned char*>(tex->pixels);

    if constexpr (sizeof(T) == 16) {
      return make_float4(float(ptr[idx * channels]), float(ptr[idx * channels + 1]), float(ptr[idx * channels + 2]), float(ptr[idx * channels + 3]));
    } else if constexpr (std::is_same_v<T, float>) {
      T result = ptr[idx * channels];
      return result;
    }
    throw std::runtime_error("Unsupported type for texture read");
  }

  template<typename T>
  ccl_device_forceinline T ccl_gpu_tex_object_read_3D(const ccl_gpu_tex_object_3D texobj,
                                                      const float fx, const float fy, const float fz)
  {
    //std::cout << "reading 3D text "  << std::endl;
    const CPUTexture3D &tex = *texobj;

    const float u = fx * tex.width  - 0.5f;
    const float v = fy * tex.height - 0.5f;
    const float w = fz * tex.depth  - 0.5f;

    const int ix = clampi(int(std::floor(u + 0.5f)), 0, tex.width  - 1);
    const int iy = clampi(int(std::floor(v + 0.5f)), 0, tex.height - 1);
    const int iz = clampi(int(std::floor(w + 0.5f)), 0, tex.depth  - 1);

    const size_t idx = ((size_t)iz * tex.height + iy) * tex.width + ix;
    const T *ptr = reinterpret_cast<const T *>(tex.voxels);

    return ptr[idx];
  }
#endif


#if defined(CPU_KERNEL) || defined(EMBREE_CPU_KERNEL) || defined(SYCL_KERNEL) || defined(ROCM_KERNEL) || defined(HIP_KERNEL) || defined(OPTIX_KERNEL)
#include "kernel/device/simple/config.h"
#include "kernel/device/simple/globals.h"
#endif

#define PRT_GLOBALS PRT_GVAR(kernel_globals, KernelParamsSimple) PRT_GVAR(warp_offset, int*)

#include <portableRT/portableRT.hpp>


#if defined(CPU_KERNEL) || defined(EMBREE_CPU_KERNEL) || defined(SYCL_KERNEL) || defined(ROCM_KERNEL) || defined(HIP_KERNEL) || defined(OPTIX_KERNEL)
#include "kernel/device/gpu/image.h"
#include "kernel/device/gpu/kernel.h"
#endif
