#pragma once


#define ccl_gpu_kernel_signature(name, ...) PRT_KERNEL(simple_##name, __VA_ARGS__)
#define ccl_gpu_kernel_postfix
#define ccl_gpu_kernel(block_num_threads, thread_num_registers)
#define ccl_gpu_kernel_threads(block_num_threads) 
#define ccl_gpu_kernel_call(x) x
#define ccl_gpu_kernel_within_bounds(i, n) ((i) < (n))


//TODO: Generalizar esto, moverlo quizás a compat
#define ccl_gpu_kernel_lambda(func, ...) \
  struct KernelLambda { \
    __VA_ARGS__; \
    __device__ int operator()(const int state) \
    { \
      return (func); \
    } \
  } ccl_gpu_kernel_lambda_pass
