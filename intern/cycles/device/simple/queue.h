#include "device/queue.h"

CCL_NAMESPACE_BEGIN

class SimpleDevice;

class SimpleDeviceQueue : public DeviceQueue {
public:
    SimpleDeviceQueue(SimpleDevice *device);
    virtual ~SimpleDeviceQueue() override;

    int num_concurrent_states(const size_t state_size) const override;
    int num_concurrent_busy_states(const size_t state_size) const override;
    void init_execution() override;
    bool enqueue(DeviceKernel kernel, const int work_size, const DeviceKernelArguments &args) override;
    bool synchronize() override;
    void zero_to_device(device_memory &mem) override;
    void copy_to_device(device_memory &mem) override;
    void copy_from_device(device_memory &mem) override;
    bool supports_local_atomic_sort() const override;
    
private:
    SimpleDevice *device;
};

CCL_NAMESPACE_END