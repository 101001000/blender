#include "device/device.h"
#include "kernel/device/simple/globals.h"
#include <portableRT/portableRT.hpp>
CCL_NAMESPACE_BEGIN

class SimpleDevice : public GPUDevice {
public:
    SimpleDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler, bool headless);
    ~SimpleDevice() override;
    BVHLayoutMask get_bvh_layout_mask(const uint kernel_features) const override;
    void const_copy_to(const char *name, void *host, const size_t size) override;
    void mem_alloc(device_memory &mem) override;
    void mem_copy_to(device_memory &mem) override;
    void mem_move_to_host(device_memory &mem) override;
    void mem_zero(device_memory &mem) override;
    void get_device_memory_info(size_t &total, size_t &free) override;
    bool alloc_device(void *&device_pointer, const size_t size) override;
    void free_device(void *device_pointer) override;
    void shared_free(void *shared_pointer) override;
    void *shared_to_device_pointer(const void *shared_pointer) override;
    void copy_host_to_device(void *device_pointer, void *host_pointer, const size_t size) override;
    void mem_copy_from(device_memory &mem, const size_t y, size_t w, const size_t h, size_t elem) override;
    void mem_free(device_memory &mem) override;
    bool shared_alloc(void *&shared_pointer, const size_t size) override;
    unique_ptr<DeviceQueue> gpu_queue_create() override;
    void build_bvh(BVH *bvh, Progress &progress, bool refit) override;
    


    void tex_alloc(device_texture &mem);
    void tex_free(device_texture &mem);
    void tex_copy_to(device_texture &mem);
    void global_alloc(device_memory &mem);
    void global_free(device_memory &mem);
    void global_copy_to(device_memory &mem);
    device_ptr mem_alloc_sub_ptr(device_memory &mem, const size_t offset, size_t /*size*/) override;

    device_vector<int> object_ids_mem;

    prt::Backend* m_backend;

};

CCL_NAMESPACE_END