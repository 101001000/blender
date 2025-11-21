#include "device/device.h"
#include "device/simple/device.h"
#include "device/simple/device_impl.h"
#include <portableRT/portableRT.hpp>

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
    info.description = "Simple Device";
    info.id = "Simple Device ID"; 
    info.num = 0;
    devices.insert(devices.begin(), info);
/*
    for (int i = 0; i < prt::available_backends().size(); i++) {
        DeviceInfo info;
        info.type = DEVICE_SIMPLE;
        info.description = prt::available_backends()[i]->name() + " - " + prt::available_backends()[i]->device_name();
        info.id = info.description; // TODO: no sé si tiene que ser único o no.
        info.num = i;
        devices.insert(devices.begin(), info);
    }*/
}

string device_simple_capabilities(){
    return "";
}

CCL_NAMESPACE_END