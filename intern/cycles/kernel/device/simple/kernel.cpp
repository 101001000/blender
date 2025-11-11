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
  #include "util/texture.h"

  float clamp_mode(float f, int mode){
    switch(mode){
      case EXTENSION_REPEAT:
         f -= floor(f);
        break;
      case EXTENSION_CLIP:
        f = f < 0.0f ? 0.0f : f > 1.0f ? 1.0f : f;
        break;
      case EXTENSION_MIRROR:
        throw std::runtime_error("Unsupported wrap type EXTENSION_MIRROR");
        break;
      case EXTENSION_EXTEND:
        throw std::runtime_error("Unsupported wrap type EXTENSION_EXTEND");
        break;
      default:
        throw std::runtime_error("Unsupported wrap type");
        break;
    }

   return f;
  }

  template<typename T>
  ccl_device_forceinline T ccl_gpu_tex_object_read_2D(const ccl_gpu_tex_object_2D texobj,
                                                      const float fx, const float fy)
  {
    const CPUTexture2D* tex = reinterpret_cast<const CPUTexture2D*>(texobj);
    float cfx = clamp_mode(fx, tex->wrap_type);
    float cfy = clamp_mode(fy, tex->wrap_type);
    
    const int ix = static_cast<int>(cfx * tex->width);
    const int iy = static_cast<int>(cfy * tex->height);
    const int channels = tex->channels;
    const int data_type = tex->data_type;

    const size_t idx = ((size_t)iy * tex->width + ix);

    switch(data_type){
      case IMAGE_DATA_TYPE_FLOAT4: {
        if constexpr (std::is_same_v<T, float4>) {
          return *reinterpret_cast<const float4*>(tex->pixels + idx * channels);
        } else if constexpr (std::is_same_v<T, float>) {
          return (*reinterpret_cast<const float4*>(tex->pixels + idx * channels)).x;
        }
      }
      case IMAGE_DATA_TYPE_BYTE4: {
        if constexpr (std::is_same_v<T, float4>) {
          uchar4 dat = *reinterpret_cast<const uchar4*>(tex->pixels + idx * channels);
          return make_float4(dat.x / 255.0f, dat.y / 255.0f, dat.z / 255.0f, dat.w / 255.0f);
        } else if constexpr (std::is_same_v<T, float>) {
          uchar4 dat = *reinterpret_cast<const uchar4*>(tex->pixels + idx * channels);
          return dat.x / 255.0f;
        }
      }
      case IMAGE_DATA_TYPE_HALF4: {
        if constexpr (std::is_same_v<T, float4>) {
          half4 dat = *reinterpret_cast<const half4*>(tex->pixels + idx * channels);
          return make_float4(dat.x, dat.y, dat.z, dat.w);
        } else if constexpr (std::is_same_v<T, float>) {
          half4 dat = *reinterpret_cast<const half4*>(tex->pixels + idx * channels);
          return dat.x;
        }
      }
      case IMAGE_DATA_TYPE_FLOAT: {
        throw std::runtime_error("Unsupported type for texture read IMAGE_DATA_TYPE_FLOAT");
      }
      case IMAGE_DATA_TYPE_BYTE: {
        throw std::runtime_error("Unsupported type for texture read IMAGE_DATA_TYPE_BYTE");
      }
      case IMAGE_DATA_TYPE_HALF: {
        if constexpr (std::is_same_v<T, float4>) {
          half dat = *reinterpret_cast<const half*>(tex->pixels + idx * channels);
          return make_float4(dat, dat, dat, dat);
        } else if constexpr (std::is_same_v<T, float>) {
          half dat = *reinterpret_cast<const half*>(tex->pixels + idx * channels);
          return dat;
        }
      }
      case IMAGE_DATA_TYPE_USHORT4: {
        throw std::runtime_error("Unsupported type for texture read IMAGE_DATA_TYPE_USHORT4");
      }
      case IMAGE_DATA_TYPE_USHORT: {
        throw std::runtime_error("Unsupported type for texture read IMAGE_DATA_TYPE_USHORT");
      }
      case IMAGE_DATA_TYPE_NANOVDB_FLOAT: {
        throw std::runtime_error("Unsupported type for texture read IMAGE_DATA_TYPE_NANOVDB_FLOAT");
      }
      case IMAGE_DATA_TYPE_NANOVDB_FLOAT3: {
        throw std::runtime_error("Unsupported type for texture read IMAGE_DATA_TYPE_NANOVDB_FLOAT3");
      }
      case IMAGE_DATA_TYPE_NANOVDB_FPN: {
        throw std::runtime_error("Unsupported type for texture read IMAGE_DATA_TYPE_NANOVDB_FPN");
      }
      case IMAGE_DATA_TYPE_NANOVDB_FP16: {
        throw std::runtime_error("Unsupported type for texture read IMAGE_DATA_TYPE_NANOVDB_FP16");
      }
    }

    throw std::runtime_error("Unsupported type for texture read");
  }

  template<typename T>
  ccl_device_forceinline T ccl_gpu_tex_object_read_3D(const ccl_gpu_tex_object_3D texobj,
                                                      const float fx, const float fy, const float fz)
  {
    throw std::runtime_error("Unsupported type for texture 3D read");
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
