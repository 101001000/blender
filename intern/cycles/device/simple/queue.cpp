#include "device/simple/queue.h"
#include "device/simple/device_impl.h"

CCL_NAMESPACE_BEGIN


SimpleDeviceQueue::SimpleDeviceQueue(SimpleDevice *device) : DeviceQueue(device), device(device) {}
SimpleDeviceQueue::~SimpleDeviceQueue() {}

int SimpleDeviceQueue::num_concurrent_states(const size_t state_size) const { return 1048576; }
int SimpleDeviceQueue::num_concurrent_busy_states(const size_t state_size) const { return 64; }
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

// Lee los primeros 8 valores de lookup_table, y después el 21759 y el 21760.
void read_lookup(SimpleDevice *device){
    char *kg_ptr = (char*)device->m_backend->get_global_ptr("kernel_globals");
  
    uintptr_t d_lookup_addr = 0;
    device->m_backend->device_copy_from(
        &d_lookup_addr,
        kg_ptr + offsetof(KernelParamsSimple, lookup_table),
        sizeof(d_lookup_addr));
  
    printf("d_lookup_addr: %p, kg_ptr %p, kg_ptroffset %p\n", (void*)d_lookup_addr, (void*)kg_ptr, (void*)(kg_ptr + offsetof(KernelParamsSimple, lookup_table)));
  
    float value;
  
    for(int i = 0; i < 8; ++i){
      device->m_backend->device_copy_from(&value, (void*)(d_lookup_addr + i * sizeof(float)), sizeof(value));
      std::cout << "value " << i << ": " << value << "\n";
    }
  
    device->m_backend->device_copy_from(&value, (void*)(d_lookup_addr + 21759 * sizeof(float)), sizeof(value));
    std::cout << "value 21759: " << value << "\n";
  
    device->m_backend->device_copy_from(&value, (void*)(d_lookup_addr + 21760 * sizeof(float)), sizeof(value));
    std::cout << "value 21760: " << value << "\n";
  }
  


bool SimpleDeviceQueue::enqueue(DeviceKernel kernel, const int work_size, const DeviceKernelArguments &args) {

    //std::cout << "Enqueueing kernel " << kernel << " with work size " << work_size << " and args size " << args.count << std::endl;

    //read_lookup(this->device);

    debug_enqueue_begin(kernel, work_size); 

    std::vector<prt::Ray> dummy_rays(work_size);
    std::vector<unsigned char> dummy_output(work_size);

    switch(kernel) {

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_INTERSECT_CLOSEST: {

        struct KernelArgs {
            int *path_index_array;
            float *render_buffer;
            int work_size;
        };

        assert(args.count == 3);
        
        KernelArgs ka;
        ka.path_index_array = get_pointer<int>(args.values[0]);
        ka.render_buffer = get_pointer<float>(args.values[1]);
        ka.work_size = get_scalar<int>(args.values[2]);
        device->m_backend->parallel_invoke("simple_integrator_intersect_closest", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_INIT_FROM_CAMERA: {
        struct KernelArgs {
            ccl::KernelWorkTile *tiles;
            int num_tiles;
            float *render_buffer;
            int max_tile_work;
        };
       
        assert(args.count == 4);

        KernelArgs ka;
        ka.tiles = get_pointer<ccl::KernelWorkTile>(args.values[0]);
        ka.num_tiles = get_scalar<int>(args.values[1]);
        ka.render_buffer = get_pointer<float>(args.values[2]);
        ka.max_tile_work = get_scalar<int>(args.values[3]);        

        device->m_backend->parallel_invoke("simple_integrator_init_from_camera", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_RESET: {
        struct KernelArgs {
            int num_states;
        };

        assert(args.count == 1);
        
        KernelArgs ka;
        ka.num_states = get_scalar<int>(args.values[0]);
        device->m_backend->parallel_invoke("simple_integrator_reset", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_PREFIX_SUM: {
        struct KernelArgs {
            int *counter;
            int *prefix_sum;
            int num_values;
        };

        assert(args.count == 3);

        KernelArgs ka;
        ka.counter = get_pointer<int>(args.values[0]);
        ka.prefix_sum = get_pointer<int>(args.values[1]);
        ka.num_values = get_scalar<int>(args.values[2]);
        device->m_backend->parallel_invoke("simple_prefix_sum", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SORTED_PATHS_ARRAY: {
        struct KernelArgs {
            int num_states;
            int num_states_limit;
            int *indices;
            int *num_indices;
            int *key_counter;
            int *key_prefix_sum;
            int kernel_index;
        };

        assert(args.count == 7);
        
        KernelArgs ka;
        ka.num_states = get_scalar<int>(args.values[0]);
        ka.num_states_limit = get_scalar<int>(args.values[1]);
        ka.indices = get_pointer<int>(args.values[2]);
        ka.num_indices = get_pointer<int>(args.values[3]);
        ka.key_counter = get_pointer<int>(args.values[4]);
        ka.key_prefix_sum = get_pointer<int>(args.values[5]);
        ka.kernel_index = get_scalar<int>(args.values[6]);
        device->m_backend->parallel_invoke("simple_integrator_sorted_paths_array", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SHADE_SURFACE: {
        struct KernelArgs {
            int *path_index_array;
            float *render_buffer;
            int work_size;
        };

        assert(args.count == 3);
        
        KernelArgs ka;
        ka.path_index_array = get_pointer<int>(args.values[0]);
        ka.render_buffer = get_pointer<float>(args.values[1]);
        ka.work_size = get_scalar<int>(args.values[2]);
        device->m_backend->parallel_invoke("simple_integrator_shade_surface", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SHADE_BACKGROUND: {
        struct KernelArgs {
            int *path_index_array;
            float *render_buffer;
            int work_size;
        };

        assert(args.count == 3);

        KernelArgs ka;
        ka.path_index_array = get_pointer<int>(args.values[0]);
        ka.render_buffer = get_pointer<float>(args.values[1]);
        ka.work_size = get_scalar<int>(args.values[2]);
        device->m_backend->parallel_invoke("simple_integrator_shade_background", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_QUEUED_PATHS_ARRAY: {
        struct KernelArgs {
            int num_states;
            int *indices;
            int *num_indices;
            int kernel_index;
        };

        assert(args.count == 4);

        KernelArgs ka;
        ka.num_states = get_scalar<int>(args.values[0]);
        ka.indices = get_pointer<int>(args.values[1]);
        ka.num_indices = get_pointer<int>(args.values[2]);
        ka.kernel_index = get_scalar<int>(args.values[3]);
        device->m_backend->parallel_invoke("simple_integrator_queued_paths_array", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_COMPACT_PATHS_ARRAY: {
        struct KernelArgs {
            int num_states;
            int *indices;
            int *num_indices;
            int num_active_paths;
        };

        assert(args.count == 4);

        KernelArgs ka;
        ka.num_states = get_scalar<int>(args.values[0]);
        ka.indices = get_pointer<int>(args.values[1]);
        ka.num_indices = get_pointer<int>(args.values[2]);
        ka.num_active_paths = get_scalar<int>(args.values[3]);
        device->m_backend->parallel_invoke("simple_integrator_compact_paths_array", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_TERMINATED_PATHS_ARRAY: {
        struct KernelArgs {
            int num_states;
            int *indices;
            int *num_indices;
            int indices_offset;
        };

        assert(args.count == 4);

        KernelArgs ka;
        ka.num_states = get_scalar<int>(args.values[0]);
        ka.indices = get_pointer<int>(args.values[1]);
        ka.num_indices = get_pointer<int>(args.values[2]);
        ka.indices_offset = get_scalar<int>(args.values[3]);
        device->m_backend->parallel_invoke("simple_integrator_terminated_paths_array", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_COMPACT_STATES: {
        struct KernelArgs {
            int *active_terminated_states;
            int active_states_offset;
            int terminated_states_offset;
            int work_size;
        };

        assert(args.count == 4);

        KernelArgs ka;
        ka.active_terminated_states = get_pointer<int>(args.values[0]);
        ka.active_states_offset = get_scalar<int>(args.values[1]);
        ka.terminated_states_offset = get_scalar<int>(args.values[2]);
        ka.work_size = get_scalar<int>(args.values[3]);
        device->m_backend->parallel_invoke("simple_integrator_compact_states", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_CHECK: {
        struct KernelArgs {
            float *render_buffer;
            int sx;
            int sy;
            int sw;
            int sh;
            float threshold;
            int reset;
            int offset;
            int stride;
            uint *num_active_pixels;
        };

        assert(args.count == 10);

        KernelArgs ka;
        ka.render_buffer = get_pointer<float>(args.values[0]);
        ka.sx = get_scalar<int>(args.values[1]);
        ka.sy = get_scalar<int>(args.values[2]);
        ka.sw = get_scalar<int>(args.values[3]);
        ka.sh = get_scalar<int>(args.values[4]);
        ka.threshold = get_scalar<float>(args.values[5]);
        ka.reset = get_scalar<int>(args.values[6]);
        ka.offset = get_scalar<int>(args.values[7]);
        ka.stride = get_scalar<int>(args.values[8]);
        ka.num_active_pixels = get_pointer<uint>(args.values[9]);
        device->m_backend->parallel_invoke("simple_adaptive_sampling_convergence_check", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SHADE_LIGHT: {
        struct KernelArgs {
            int *path_index_array;
            float *render_buffer;
            int work_size;
        };

        assert(args.count == 3);

        KernelArgs ka;
        ka.path_index_array = get_pointer<int>(args.values[0]);
        ka.render_buffer = get_pointer<float>(args.values[1]);
        ka.work_size = get_scalar<int>(args.values[2]);
        device->m_backend->parallel_invoke("simple_integrator_shade_light", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_X: {
        struct KernelArgs {
            float *render_buffer;
            int sx, sy, sw, sh;
            int offset;
            int stride;
        };

        assert(args.count == 7);

        KernelArgs ka;
        ka.render_buffer = get_pointer<float>(args.values[0]);
        ka.sx = get_scalar<int>(args.values[1]);
        ka.sy = get_scalar<int>(args.values[2]);
        ka.sw = get_scalar<int>(args.values[3]);
        ka.sh = get_scalar<int>(args.values[4]);
        ka.offset = get_scalar<int>(args.values[5]);
        ka.stride = get_scalar<int>(args.values[6]);
        device->m_backend->parallel_invoke("simple_adaptive_sampling_filter_x", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }

    case DeviceKernel::DEVICE_KERNEL_ADAPTIVE_SAMPLING_CONVERGENCE_FILTER_Y: {
        struct KernelArgs {
            float *render_buffer;
            int sx, sy, sw, sh;
            int offset;
            int stride;
        };

        assert(args.count == 7);

        KernelArgs ka;
        ka.render_buffer = get_pointer<float>(args.values[0]);
        ka.sx = get_scalar<int>(args.values[1]);
        ka.sy = get_scalar<int>(args.values[2]);
        ka.sw = get_scalar<int>(args.values[3]);
        ka.sh = get_scalar<int>(args.values[4]);
        ka.offset = get_scalar<int>(args.values[5]);
        ka.stride = get_scalar<int>(args.values[6]);
        device->m_backend->parallel_invoke("simple_adaptive_sampling_filter_y", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }
    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_INTERSECT_SHADOW: {

        struct KernelArgs {
            int *path_index_array;
            int work_size;
        };

        assert(args.count == 2);

        KernelArgs ka;
        ka.path_index_array = get_pointer<int>(args.values[0]);
        ka.work_size = get_scalar<int>(args.values[1]);
        device->m_backend->parallel_invoke("simple_integrator_intersect_shadow", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;

    }
    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_SHADE_SHADOW:{
        struct KernelArgs {
            int *path_index_array;
            float* render_buffer;
            int work_size;
        };

        assert(args.count == 3);

        KernelArgs ka;
        ka.path_index_array = get_pointer<int>(args.values[0]);
        ka.render_buffer = get_pointer<float>(args.values[1]);
        ka.work_size = get_scalar<int>(args.values[2]);
        device->m_backend->parallel_invoke("simple_integrator_shade_shadow", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }
    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_QUEUED_SHADOW_PATHS_ARRAY: {
        struct KernelArgs {
            int num_states;
            int *indices;
            int *num_indices;
            int kernel_index;
        };

        assert(args.count == 4);

        KernelArgs ka;
        ka.num_states = get_scalar<int>(args.values[0]);
        ka.indices = get_pointer<int>(args.values[1]);
        ka.num_indices = get_pointer<int>(args.values[2]);
        ka.kernel_index = get_scalar<int>(args.values[3]);
        device->m_backend->parallel_invoke("simple_integrator_queued_shadow_paths_array", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }
    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_TERMINATED_SHADOW_PATHS_ARRAY: {
        struct KernelArgs {
            int num_states;
            int *indices;
            int *num_indices;
            int indices_offset;
        };

        assert(args.count == 4);

        KernelArgs ka;
        ka.num_states = get_scalar<int>(args.values[0]);
        ka.indices = get_pointer<int>(args.values[1]);
        ka.num_indices = get_pointer<int>(args.values[2]);
        ka.indices_offset = get_scalar<int>(args.values[3]);
        device->m_backend->parallel_invoke("simple_integrator_terminated_shadow_paths_array", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }    
    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_PATHS_ARRAY: {
        struct KernelArgs {
            int num_states;
            int *indices;
            int *num_indices;
            int num_active_paths;
        };

        assert(args.count == 4);

        KernelArgs ka;
        ka.num_states = get_scalar<int>(args.values[0]);
        ka.indices = get_pointer<int>(args.values[1]);
        ka.num_indices = get_pointer<int>(args.values[2]);
        ka.num_active_paths = get_scalar<int>(args.values[3]);
        device->m_backend->parallel_invoke("simple_integrator_compact_shadow_paths_array", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }
    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_COMPACT_SHADOW_STATES: {
        struct KernelArgs {
            int *active_terminated_states;
            int active_states_offset;
            int terminated_states_offset;
            int work_size;
        };

        assert(args.count == 4);

        KernelArgs ka;
        ka.active_terminated_states = get_pointer<int>(args.values[0]);
        ka.active_states_offset = get_scalar<int>(args.values[1]);
        ka.terminated_states_offset = get_scalar<int>(args.values[2]);
        ka.work_size = get_scalar<int>(args.values[3]);
        device->m_backend->parallel_invoke("simple_integrator_compact_shadow_states", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }
    case DeviceKernel::DEVICE_KERNEL_SHADER_EVAL_BACKGROUND: {
        struct KernelArgs {
            KernelShaderEvalInput *input;
            float *output;
            int offset;
            int work_size;
        };

        assert(args.count == 4);

        KernelArgs ka;
        ka.input = get_pointer<KernelShaderEvalInput>(args.values[0]);
        ka.output = get_pointer<float>(args.values[1]);
        ka.offset = get_scalar<int>(args.values[2]);
        ka.work_size = get_scalar<int>(args.values[3]);
        device->m_backend->parallel_invoke("simple_shader_eval_background", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }
    case DeviceKernel::DEVICE_KERNEL_SHADER_EVAL_CURVE_SHADOW_TRANSPARENCY: {
        struct KernelArgs {
            KernelShaderEvalInput *input;
            float *output;
            int offset;
            int work_size;
        };

        assert(args.count == 4);

        KernelArgs ka;
        ka.input = get_pointer<KernelShaderEvalInput>(args.values[0]);
        ka.output = get_pointer<float>(args.values[1]);
        ka.offset = get_scalar<int>(args.values[2]);
        ka.work_size = get_scalar<int>(args.values[3]);
        device->m_backend->parallel_invoke("simple_shader_eval_curve_shadow_transparency", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }
    case DeviceKernel::DEVICE_KERNEL_INTEGRATOR_INTERSECT_SUBSURFACE: {
        struct KernelArgs {
            int *path_index_array;
            int work_size;
        };

        assert(args.count == 2);

        KernelArgs ka;
        ka.path_index_array = get_pointer<int>(args.values[0]);
        ka.work_size = get_scalar<int>(args.values[1]);
        device->m_backend->parallel_invoke("simple_integrator_intersect_subsurface", dummy_rays, dummy_output, &ka, sizeof(ka));
        break;
    }
    default:
        std::cout << "unknown kernel " << device_kernel_as_string(kernel) << std::endl;
        break;
    }

    debug_enqueue_end();
    return true;
}
bool SimpleDeviceQueue::synchronize() { return true; }
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