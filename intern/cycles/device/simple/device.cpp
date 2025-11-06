#include "device/device.h"
#include "device/simple/device.h"
#include "device/simple/device_impl.h"

CCL_NAMESPACE_BEGIN

bool device_simple_init() {
    return true;
}

unique_ptr<Device> device_simple_create(const DeviceInfo &info,
    Stats &stats,
    Profiler &profiler,
    bool headless){
    return make_unique<SimpleDevice>(info, stats, profiler, headless);
}

void device_simple_info(vector<DeviceInfo> &devices){
    DeviceInfo info;
    info.type = DEVICE_SIMPLE;
    info.description = "Simple description";
    info.id = "Simple ID";
    info.num = 0;
    devices.insert(devices.begin(), info);
}

string device_simple_capabilities(){
    return "";
}

CCL_NAMESPACE_END