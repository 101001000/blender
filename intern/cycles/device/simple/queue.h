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
    std::size_t m_concurrent_states;
    std::size_t m_concurrent_busy_states;
    SimpleDevice *device;
    std::map<std::string, std::map<int, std::size_t>> kernel_blocksize;
};

CCL_NAMESPACE_END