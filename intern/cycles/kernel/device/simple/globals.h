#pragma once

#include <iostream>

#include "kernel/types.h"

#include "kernel/integrator/state.h"
#include "kernel/util/profiler.h"

#include "util/color.h"
#include "util/texture.h"

#if defined(PRT_SYCL_KERNEL) || defined(PRT_EMBREE_SYCL_KERNEL)
#include <sycl/sycl.hpp>
#endif


CCL_NAMESPACE_BEGIN

struct IntegratorStateGPU;

struct KernelGlobalsGPU {
  int unused[1];
};


using KernelGlobals = ccl_global __attribute__((unused)) KernelGlobalsGPU *ccl_restrict;

#define kernel_data (g_data)
#define kernel_data_fetch(name, index) (g_##name[index])
#define kernel_data_array(name) (g_##name)
#define kernel_integrator_state (g_integrator_state)


CCL_NAMESPACE_END