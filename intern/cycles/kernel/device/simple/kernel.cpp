#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wmissing-declarations"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#pragma GCC diagnostic ignored "-Wreorder"

// TODO: Limpiar esto
#if defined(CPU_KERNEL) || defined(EMBREE_CPU_KERNEL) || defined(SYCL_KERNEL) || defined(OPTIX_KERNEL)
  #include "kernel/device/simple/compat.h"
  #include "util/half.h"
  #include "util/types.h"
  #include "util/texture.h"

  ccl_device_forceinline float clamp_mode(float f, int mode){
    switch(mode){
      case EXTENSION_REPEAT:
         f -= floor(f);
        break;
      case EXTENSION_CLIP:
        f = f < 0.0f ? 0.0f : f > 1.0f ? 1.0f : f;
        break;
      case EXTENSION_MIRROR:
        //throw std::runtime_error("Unsupported wrap type EXTENSION_MIRROR");
        break;
      case EXTENSION_EXTEND:
        //throw std::runtime_error("Unsupported wrap type EXTENSION_EXTEND");
        break;
      default:
        //throw std::runtime_error("Unsupported wrap type");
        break;
    }

   return f;
  }

  template<typename T>
  ccl_device_forceinline T ccl_gpu_tex_object_read_2D(const ccl_gpu_tex_object_2D texobj,
                                                      float fx, float fy)
  {
    int a;
    const CPUTexture2D* tex = reinterpret_cast<const CPUTexture2D*>(texobj);
  
    const float cfx = clamp_mode(fx, tex->wrap_type);
    const float cfy = clamp_mode(fy, tex->wrap_type);
  
    const int ix = static_cast<int>(cfx * tex->width);
    const int iy = static_cast<int>(cfy * tex->height);
    const int channels = tex->channels;
    const int data_type = tex->data_type;
    const size_t idx = ((size_t)iy * tex->width + ix);
  
    switch (data_type) {
      case IMAGE_DATA_TYPE_FLOAT4: {
        const float4 v = reinterpret_cast<const float4*>(tex->pixels)[idx];
        if constexpr (std::is_same_v<T,float4>) return v;
        if constexpr (std::is_same_v<T,float>)  return v.x;
        break;
      }
      case IMAGE_DATA_TYPE_BYTE4: {
        const uchar4 u = reinterpret_cast<const uchar4*>(tex->pixels)[idx];
        if constexpr (std::is_same_v<T,float4>) return make_float4(u.x/255.f,u.y/255.f,u.z/255.f,u.w/255.f);
        if constexpr (std::is_same_v<T,float>)  return u.x/255.f;
        break;
      }
      case IMAGE_DATA_TYPE_HALF4: {
        //throw std::runtime_error("Unsupported type for texture read IMAGE_DATA_TYPE_HALF4");
        break;
      }

      case IMAGE_DATA_TYPE_HALF: {
        //throw std::runtime_error("Unsupported type for texture read IMAGE_DATA_TYPE_HALF");
        break;
      }
      case IMAGE_DATA_TYPE_USHORT4:{
        const ushort4 v = reinterpret_cast<const ushort4*>(tex->pixels)[idx];
        if constexpr (std::is_same_v<T,float4>) return make_float4(v.x/65535.0f,v.y/65535.0f,v.z/65535.0f,v.w/65535.0f);
        if constexpr (std::is_same_v<T,float>)  return v.x/65535.0f;
        break;
      }
      case IMAGE_DATA_TYPE_USHORT:{
        const ushort v = reinterpret_cast<const ushort*>(tex->pixels)[idx];
        if constexpr (std::is_same_v<T,float4>) return make_float4(v/65535.0f,v/65535.0f,v/65535.0f,v/65535.0f);
        if constexpr (std::is_same_v<T,float>)  return v/65535.0f;
        break;
      }
      case IMAGE_DATA_TYPE_FLOAT:
      case IMAGE_DATA_TYPE_BYTE:
      case IMAGE_DATA_TYPE_NANOVDB_FLOAT:
      case IMAGE_DATA_TYPE_NANOVDB_FLOAT3:
      case IMAGE_DATA_TYPE_NANOVDB_FPN:
      case IMAGE_DATA_TYPE_NANOVDB_FP16:
        break;
    }
    return T();
    //throw std::runtime_error("Unsupported type for texture read");
  }

  template<typename T>
  ccl_device_forceinline T ccl_gpu_tex_object_read_3D(const ccl_gpu_tex_object_3D texobj,
                                                      const float fx, const float fy, const float fz)
  {
    //throw std::runtime_error("Unsupported type for texture 3D read");
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

#pragma GCC diagnostic pop