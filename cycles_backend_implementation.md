# Cycles Backend Implementation

A Cycles backend can be divided into two main parts: the **host implementation**, which defines how Cycles manages and submits work to a device, and the **device implementation**, which adapts the common Cycles kernel code to the execution model of the backend.

## Host Implementation

### [`device_impl.cpp`](intern/cycles/device/simple/device_impl.cpp)

Defines how the device is managed from the host side. It implements the operations expected by Cycles for device initialization, memory management, global data updates, and acceleration structure setup.

| Responsibility | CUDA example | PortableRT example |
|---|---|---|
| Memory handling | `cuMemAlloc(...)`, `cuMemcpyHtoD(...)`, etc. | `device_malloc(...)`, `device_copy_to(...)`, etc. |
| Global / constant data | `cuModuleGetGlobal(...)`, `cuMemcpyHtoD(...)` | `get_global_ptr(...)`, `const_copy_to(...)` |
| Device setup | CUDA device/context setup | `kernelapi_init(...)`, `select_backend(...)` |
| BVH setup | CUDA/Cycles BVH setup | `backend.set_tris(...)` |

### [`queue.cpp`](intern/cycles/device/simple/queue.cpp)

Defines how work is submitted to the device from the host side. It implements the queue abstraction used by Cycles to configure kernel execution, invoke kernels, and synchronize the backend.

| Responsibility | CUDA example | PortableRT example |
|---|---|---|
| Kernel configuration | `kernel<<<grid_dim, block_dim>>>(...)` | `set_ka_blocksize(...)` |
| Kernel invocation | `kernel<<<...>>>(...)` | `backend.parallel_invoke_async(...)` |
| Synchronization | `cudaStreamSynchronize(...)` | `backend.sync()` |

## Device Implementation

Most Cycles device code is shared between backends. A backend provides the primitives, execution model, global data representation, and ray-intersection interface expected by those common kernels.

### [`bvh.h`](intern/cycles/kernel/device/simple/bvh.h)

Defines the scene-intersection interface used by the Cycles kernels and maps it to the traversal mechanism provided by the backend.

| Responsibility | OptiX example | PortableRT example |
|---|---|---|
| Scene traversal | `optixTrace(...)` | `prt::closest_hit(...)` |
| Hit filtering | Any-hit programs | Cycles-side filtering |
| Hit data | OptiX payloads | `prt::Hit` |

In PortableRT, a Cycles ray is converted to `prt::Ray`, traversal is performed through `prt::closest_hit(...)`, and the returned primitive and instance information is mapped back to the Cycles representation. Visibility, self-intersection, and geometry-skipping rules remain part of the Cycles-side intersection logic.

### Kernel Device Implementation Headers

The common device kernels are implemented in shared headers such as [`kernel.h`](intern/cycles/kernel/device/gpu/kernel.h) and [`image.h`](intern/cycles/kernel/device/gpu/image.h).

These headers use macros and compile-time definitions to select backend-specific implementations and optimizations. For example, an operation such as `count_leading_zeros()` can resolve to CUDA/HIP `__clz`, SYCL `sycl::clz`, or a generic implementation.

### [`compat.h`](intern/cycles/kernel/device/simple/compat.h)

Defines the macros and low-level primitives expected by the common Cycles kernel headers. Each backend maps these abstractions to its own execution model, including function qualifiers, synchronization, indexing, atomics, mathematical intrinsics, and kernel declarations.

For example, common kernel code can use:

```cpp
ccl_gpu_block_idx_x
```

while CUDA can define:

```cpp
#define ccl_gpu_block_idx_x blockIdx.x
```

and PortableRT can map the same abstraction to:

```cpp
#define ccl_gpu_block_idx_x global_idx
```

This allows the common kernel implementation to remain independent of how a backend represents threads or work-items.

### [`globals.h`](intern/cycles/kernel/device/simple/globals.h)

Defines how the global state expected by Cycles kernels is represented and accessed.

PortableRT stores this state in `KernelParamsSimple`, while abstractions such as `kernel_data`, `kernel_data_fetch(...)`, and `kernel_integrator_state` expose it using the interface expected by the common kernel code.

### [`kernel.cpp`](intern/cycles/kernel/device/simple/kernel.cpp)

Acts as the device compilation entry point. It defines the backend global state and includes the common Cycles GPU kernel implementation.

The backend-specific definitions provided by `compat.h` and `globals.h` determine how that common kernel code is compiled for the selected target.
