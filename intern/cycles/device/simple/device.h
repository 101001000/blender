#pragma once

#include "util/unique_ptr.h"
#include "util/vector.h"

CCL_NAMESPACE_BEGIN
 
class Device;
class DeviceInfo;
class Profiler;
class Stats;
 
bool device_simple_init();
 
unique_ptr<Device> device_simple_create(const DeviceInfo &info,
                                       Stats &stats,
                                       Profiler &profiler,
                                       bool headless);
 
void device_simple_info(vector<DeviceInfo> &devices);
 
string device_simple_capabilities();
 
CCL_NAMESPACE_END
 