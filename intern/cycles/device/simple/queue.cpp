#include <cstdlib> 

#include "device/simple/queue.h"
#include "device/simple/device_impl.h"
#include "kernel/device/simple/globals.h"
#include "util/defines.h"
#include <cstring>


CCL_NAMESPACE_BEGIN

SimpleDeviceQueue::SimpleDeviceQueue(SimpleDevice *device) : DeviceQueue(device), device(device) {


    const int max_num_threads = device->m_backend->device_compute_units() * device->m_backend->device_max_threads_per_compute_unit();

    if(device->m_backend->name() == "EMBREE_SYCL" || device->m_backend->name() == "SYCL"){
        m_concurrent_states = 16 * max(8 * max_num_threads, 65536);
        m_concurrent_busy_states = 4 * max(8 * max_num_threads, 65536);
    } else {
        m_concurrent_states = max(max_num_threads, 65536) * 16;
        m_concurrent_busy_states = 4 * max_num_threads;
    }

    if (const char* env = std::getenv("CYCLES_CONCURRENT_STATES")) {
        char* end = nullptr;
        unsigned long v = std::strtoul(env, &end, 10);

        if (end != env && *end == '\0' && v > 0) {
            m_concurrent_states = static_cast<size_t>(v);
        }
    }
    
    if (const char* env = std::getenv("CYCLES_CONCURRENT_BUSY_STATES")) {
        char* end = nullptr;
        unsigned long v = std::strtoul(env, &end, 10);

        if (end != env && *end == '\0' && v > 0) {
            m_concurrent_busy_states = static_cast<size_t>(v);
        }
    }

    
    for(int i = 0; i < DeviceKernel::DEVICE_KERNEL_NUM; i++){
        kernel_blocksize["CPU"][i] = 64;
        kernel_blocksize["OPTIX"][i] = 512;
        kernel_blocksize["HIP"][i] = 1024;
        kernel_blocksize["EMBREE_SYCL"][i] = 512;
        kernel_blocksize["EMBREE_CPU"][i] = 64;
        if(device->m_backend->device_name().find("CUDA") != std::string::npos || device->m_backend->device_name().find("770") != std::string::npos){
            kernel_blocksize["SYCL"][i] = 512;
        } else { // AMD
            kernel_blocksize["SYCL"][i] = 512;
        }
    }

    kernel_blocksize["OPTIX"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA] = 448;


    if(device->m_backend->device_name().find("770") != std::string::npos){
        std::cout << "ARC LAYOUT" << std::endl;
        kernel_blocksize["SYCL"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_RESET] = 1024;

        kernel_blocksize["SYCL"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE] = 64;
        kernel_blocksize["SYCL"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW] = 64;
        kernel_blocksize["SYCL"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT] = 64;
        kernel_blocksize["SYCL"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND] = 64;

        kernel_blocksize["SYCL"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST] = 128;
        kernel_blocksize["SYCL"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW] = 128;
        kernel_blocksize["SYCL"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA] = 128;
    } else if(device->m_backend->device_name().find("CUDA") != std::string::npos) {
        
        std::cout << "CUDA LAYOUT" << std::endl;
    } else {
        std::cout << "HIP LAYOUT" << std::endl;
    }


    //kernel_blocksize["HIP"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST] = 256;
    //kernel_blocksize["HIP"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW] = 256;
    //kernel_blocksize["HIP"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA] = 256;

    kernel_blocksize["EMBREE_SYCL"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_RESET] = 1024;

    kernel_blocksize["EMBREE_SYCL"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE] = 64;
    kernel_blocksize["EMBREE_SYCL"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW] = 64;
    kernel_blocksize["EMBREE_SYCL"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT] = 64;
    kernel_blocksize["EMBREE_SYCL"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND] = 64;

    kernel_blocksize["EMBREE_SYCL"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST] = 128;
    kernel_blocksize["EMBREE_SYCL"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW] = 128;
    kernel_blocksize["EMBREE_SYCL"][DeviceKernel::DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA] = 128;

    //std::cout << "CYCLES_CONCURRENT_STATES: " << m_concurrent_states << "\n";
    //std::cout << "CYCLES_CONCURRENT_BUSY_STATES: " << m_concurrent_busy_states << "\n";
}
SimpleDeviceQueue::~SimpleDeviceQueue() {}

int SimpleDeviceQueue::num_concurrent_states(const size_t state_size) const {
    
    VLOG_DEVICE_STATS << "GPU queue concurrent states: " << m_concurrent_states << ", using up to " << string_human_readable_size(m_concurrent_states * state_size);
    
    return m_concurrent_states;
}
int SimpleDeviceQueue::num_concurrent_busy_states(const size_t state_size) const {
    VLOG_DEVICE_STATS << "GPU queue concurrent busy states: " << m_concurrent_busy_states;
    return m_concurrent_busy_states;
}
void SimpleDeviceQueue::init_execution() {
    debug_init_execution();
    device->load_texture_info();
}

template<typename T>
inline T get_scalar(const void *p)
{
    return *reinterpret_cast<const T *>(p);
}
template<typename T>
inline T *get_pointer(const void *p)
{
  return *reinterpret_cast<T * const *>(p);
}

std::string type_to_string(DeviceKernelArguments::Type type) {
    switch(type) {
        case DeviceKernelArguments::Type::POINTER: return "pointer";
        case DeviceKernelArguments::Type::INT32: return "int32";
        case DeviceKernelArguments::Type::FLOAT32: return "float32";
        case DeviceKernelArguments::Type::KERNEL_FILM_CONVERT: return "kernel_film_convert";
        case DeviceKernelArguments::Type::HIPRT_GLOBAL_STACK: return "hiprt_global_stack";
        default: return "unknown";
    }
}

size_t arg_alignment(const DeviceKernelArguments::Type type)
{
  switch (type) {
    case DeviceKernelArguments::Type::POINTER:
      return alignof(device_ptr);
    case DeviceKernelArguments::Type::INT32:
      return alignof(int32_t);
    case DeviceKernelArguments::Type::FLOAT32:
      return alignof(float);
    case DeviceKernelArguments::Type::KERNEL_FILM_CONVERT:
      return alignof(KernelFilmConvert);
    case DeviceKernelArguments::Type::HIPRT_GLOBAL_STACK:
      return alignof(void *);
    default:
      assert(false);
      return 1;
  }
}

std::vector<std::byte> pack_kernel_args(const DeviceKernelArguments &args)
{
  size_t offset = 0;
  size_t max_alignment = 1;

  std::array<size_t, DeviceKernelArguments::MAX_ARGS> offsets{};

  for (size_t i = 0; i < args.count; ++i) {
    const size_t alignment = arg_alignment(args.types[i]);

    offset = align_up(offset, alignment);
    offsets[i] = offset;

    offset += args.sizes[i];
    max_alignment = std::max(max_alignment, alignment);
  }

  const size_t total_size = align_up(offset, max_alignment);

  std::vector<std::byte> packed_args(total_size);

  for (size_t i = 0; i < args.count; ++i) {
    std::memcpy(packed_args.data() + offsets[i], args.values[i], args.sizes[i]);
  }

  return packed_args;
}

bool SimpleDeviceQueue::enqueue(DeviceKernel kernel, const int work_size, const DeviceKernelArguments &args) {

    auto start = std::chrono::high_resolution_clock::now();

    debug_enqueue_begin(kernel, work_size); 

    if(device_kernel_has_intersection(kernel)){
        device->m_backend->m_generic_kernel = false;
    } else {
        device->m_backend->m_generic_kernel = true;
    }

    std::size_t blocksize = kernel_blocksize[device->m_backend->name()][static_cast<int>(kernel)];
    device->m_backend->set_ka_blocksize(blocksize); 


    std::cout << "Enqueueing kernel " << kernel << " with work size " << work_size << " and blocksize " << blocksize << std::endl;

    if(kernel == DeviceKernel::DEVICE_KERNEL_SHADER_EVAL_DISPLACE) {
      return true;
    }

    std::size_t dummy_rays = work_size; // TODO: cleanup
    std::vector<unsigned char> dummy_output(0);

    const std::vector<std::byte> packed_args = pack_kernel_args(args);

    const std::string kernel_name =
    std::string("simple_") + device_kernel_as_string(kernel);

    device->m_backend->parallel_invoke_async(
        kernel_name.c_str(),
        dummy_rays,
        dummy_output,
        packed_args.data(),
        packed_args.size());

    debug_enqueue_end();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    if( device->kernel_times.find(device_kernel_as_string(kernel)) == device->kernel_times.end() ) {
        device->kernel_times[device_kernel_as_string(kernel)] = duration;
    } else {
        device->kernel_times[device_kernel_as_string(kernel)] += duration;
    }

    return true;
}
bool SimpleDeviceQueue::synchronize() { 
  device->m_backend->sync();
  return true;
}
void SimpleDeviceQueue::zero_to_device(device_memory &mem) {device->mem_zero(mem);}
void SimpleDeviceQueue::copy_to_device(device_memory &mem) {

    //std::cout << "queue copy_to_device " << mem.name << std::endl;
  assert(mem.type != MEM_GLOBAL && mem.type != MEM_TEXTURE);

  if (mem.memory_size() == 0) {
    return;
  }

  /* Allocate on demand. */
  if (mem.device_pointer == 0) {
    device->mem_alloc(mem);
  }

  assert(mem.device_pointer != 0);
  assert(mem.host_pointer != nullptr);

  device->m_backend->device_copy_to((char *)mem.device_pointer, (char *)mem.host_pointer, mem.memory_size());

}
void SimpleDeviceQueue::copy_from_device(device_memory &mem) {

    //std::cout << "queue copy_from_device " << mem.name << std::endl;
  assert(mem.type != MEM_GLOBAL && mem.type != MEM_TEXTURE);

  if (mem.memory_size() == 0) {
    return;
  }

  assert(mem.device_pointer != 0);
  assert(mem.host_pointer != nullptr);

  device->m_backend->device_copy_from((char *)mem.host_pointer, (char *)mem.device_pointer, mem.memory_size());

}
bool SimpleDeviceQueue::supports_local_atomic_sort() const { return false; }

CCL_NAMESPACE_END