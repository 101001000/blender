#pragma once

#include <iostream>

#include "kernel/types.h"

#include "kernel/integrator/state.h"
#include "kernel/util/profiler.h"

#include "util/color.h"
#include "util/texture.h"

CCL_NAMESPACE_BEGIN

struct IntegratorStateGPU;

struct KernelGlobalsGPU {
  int unused[1];
};

struct KernelParamsSimple {

    #define KERNEL_DATA_ARRAY(type, name) const type *name = nullptr;
    KERNEL_DATA_ARRAY(int, object_ids)
    #include "kernel/data_arrays.h"
    #undef KERNEL_DATA_ARRAY
      KernelData data;
      IntegratorStateGPU integrator_state;
};

using KernelGlobals = ccl_global KernelGlobalsGPU *ccl_restrict;

template<typename T>
ccl_device_inline const T &kernel_data_fetch_dbg_ref(const char *nm,
                                                    const T     *base,
                                                    size_t       i,
                                                    void     *base2)
{
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

#define kernel_data (get_global_value(KernelParamsSimple, kernel_globals).data)
#define kernel_data_fetch(name, index) \
  kernel_data_fetch_dbg_ref(#name, get_global_value(KernelParamsSimple, kernel_globals).name, (index), &get_global_value(KernelParamsSimple, kernel_globals))
#define kernel_data_array(name) (get_global_value(KernelParamsSimple, kernel_globals).name)
#define kernel_integrator_state (get_global_value(KernelParamsSimple, kernel_globals).integrator_state)

CCL_NAMESPACE_END