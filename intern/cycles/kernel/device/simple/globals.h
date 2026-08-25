#pragma once

#include <iostream>

#include "kernel/types.h"

#include "kernel/integrator/state.h"
#include "kernel/util/profiler.h"

#include "util/color.h"
#include "util/texture.h"

#if defined(PRT_KERNEL_SYCL) || defined(PRT_KERNEL_EMBREE_SYCL)
#include <sycl/sycl.hpp>
#endif

CCL_NAMESPACE_BEGIN

struct IntegratorStateGPU;

struct KernelGlobalsGPU {
  int unused[1];
};

struct KernelParamsSimple {

    #define KERNEL_DATA_ARRAY(type, name) const type *name = nullptr;
    KERNEL_DATA_ARRAY(int, object_ids)
    KERNEL_DATA_ARRAY(int, prim_ids)
    KERNEL_DATA_ARRAY(int, object_sizes)
    KernelData data;
    IntegratorStateGPU integrator_state;
    #include "kernel/data_arrays.h"
    #undef KERNEL_DATA_ARRAY

};

using KernelGlobals = ccl_global KernelGlobalsGPU *ccl_restrict;

template<typename T>
ccl_device_inline const T &kernel_data_fetch_dbg_ref(const char *nm,
                                                    const T     *base,
                                                    size_t       i,
                                                    void     *base2)
{
  if(i == static_cast<size_t>(-1)){
    //throw std::runtime_error("Invalid index");
    //
    #ifdef PRT_KERNEL_HIP
      //printf("Invalid index %s %p %d\n", nm, base, i);
    #endif
    #ifdef PRT_KERNEL_SYCL
    //sycl::ext::oneapi::experimental::printf("Invalid index access for %s\n", nm);
    #endif
    #ifdef PRT_KERNEL_CPU
      throw std::runtime_error("Invalid index access for " + std::string(nm));
    #endif
    return base[0];
  }
  //printf("kernel_data_fetch_dbg_ref %s %p %d\n", nm, base, i);
  //printf("kernel_data_fetch_dbg_ref end %s %p %d\n", nm, base, i);
  /*
  if(((uintptr_t)base) == 0xb02e63600){
    if constexpr(std::is_same<T, float>::value){
      printf("Dumping all 0xb02e63600 values: ");
      for(int i = 0; i < 8; i++){
        printf("%d %f \n", i, (double)base[i]);
      }
      printf("%d %f \n", 21759, (double)base[21759]);
      printf("%d %f \n", 21760, (double)base[21760]);
      printf("\n");
    }
  }*/
  
  if constexpr(std::is_same<T, int>::value){
    //printf("kernel_data_fetch_dbg_ref %s %p %zu %p %d\n", nm, base, i, base2, (int)base[i]);
  }else if constexpr(std::is_same<T, float>::value){
    //printf("kernel_data_fetch_dbg_ref %s %p %f %f\n", nm, base, (double)base[i], (double)base[0]);
  }else{
    //printf("kernel_data_fetch_dbg_ref %s %p %zu %p\n", nm, base, i, base2);
  }
  //printf("kernel_data_fetch_dbg_ref %s %p %zu %p\n", nm, base, i, base2);
  /*
  if(static_cast<int>(i) < 0){
    std::cout << "WRONG ACCESS [device] " << nm << " = " << static_cast<const void *>(base) << " idx " << i
            << std::endl;
    return base[0];
  }*/
  //std::cout << "[device] " << nm << " = " << static_cast<const void *>(base) << " idx " << i
  //          << std::endl;
  return base[i];            
}

#define kernel_data (kernel_params_simple.data)
#define kernel_data_fetch(name, index) \
  kernel_data_fetch_dbg_ref(#name, kernel_params_simple.name, (index), &kernel_params_simple)
#define kernel_data_array(name) (kernel_params_simple.name)
#define kernel_integrator_state (kernel_params_simple.integrator_state)

CCL_NAMESPACE_END