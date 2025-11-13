#include <sys/sysinfo.h>
#include "device/simple/device_impl.h"
#include "kernel/device/simple/globals.h"
#include "device/simple/queue.h"
#include "bvh/bvh2.h"
#include <portableRT/portableRT.hpp>
#include "scene/geometry.h"
#include "scene/object.h"
#include "scene/mesh.h"
#include <dlfcn.h>
#include <cinttypes>

CCL_NAMESPACE_BEGIN

SimpleDevice::SimpleDevice(const DeviceInfo &info, Stats &stats, Profiler &profiler, bool headless) : GPUDevice(info, stats, profiler, headless), object_ids_mem(this, "object_ids", MEM_GLOBAL), prim_ids_mem(this, "prim_ids", MEM_GLOBAL), object_sizes_mem(this, "object_sizes", MEM_GLOBAL) {

    prt::kernelapi_init({{"kernel_globals", sizeof(KernelParamsSimple)}, {"warp_offset", sizeof(int*)}});

    std::cout << "available backends: " << std::endl;
    for (int i = 0; i < prt::available_backends().size(); i++) {
        std::cout << i << ": " << prt::available_backends()[i]->name() << " - " << prt::available_backends()[i]->device_name() << " " << std::endl;
    }
    std::cout << "select backend: ";
    int idx;
    //std::cin >> idx;
    idx = 1;
    prt::select_backend(prt::available_backends()[idx]);
    m_backend = prt::selected_backend;
    std::cout << "selected backend: " << prt::selected_backend->name() << std::endl;
    m_backend->global_alloc("kernel_globals", sizeof(KernelParamsSimple));
    m_backend->global_alloc("warp_offset", sizeof(int*));
    void* warp_offset = m_backend->device_malloc(sizeof(int) * (1024 + 1));
    m_backend->global_copy_to("warp_offset", &warp_offset, sizeof(int*));

    
    
    /*
    std::cout << "showing all kernels:" << std::endl;
    for (auto [name, fn] : prt::kernels_) {
        //std::cout << "Listing kernel " << name << std::endl;
    }

    for (auto [data, size] : prt::embeded_kernels_) {
        //std::cout << "Listing embeded kernel " << size << std::endl;
    }

    std::cout << "test" << std::endl;

    for (auto [name, size] : prt::global_vars_) {
        std::cout << "Listing global " << name << " of size " << size << std::endl;
    }*/

}

SimpleDevice::~SimpleDevice() {
    texture_info.free();
    m_backend->global_free("kernel_globals");
}

void SimpleDevice::global_free(device_memory &mem)
{
  //check
  if (mem.is_resident(this) && mem.device_pointer) {
    generic_free(mem);
  }
}

void SimpleDevice::tex_alloc(device_texture &mem)
{

    auto cmem = generic_alloc(mem);
    if (!cmem) {
      return;
    }

    m_backend->device_copy_to((void*)mem.device_pointer, mem.host_pointer, mem.memory_size());

    struct CPUTexture2D {
      const void *pixels;
      int         width;
      int         height;
      int         channels;  
      int         data_type;
      int         wrap_type;
    };

    CPUTexture2D dev_tex_info;
    dev_tex_info.pixels = reinterpret_cast<const void*>(mem.device_pointer);
    dev_tex_info.width = mem.info.width;
    dev_tex_info.height = mem.info.height;
    dev_tex_info.channels = mem.data_elements;
    dev_tex_info.data_type = mem.info.data_type;
    dev_tex_info.wrap_type = mem.info.extension;
    
    
    void* tex_info_ptr = m_backend->device_malloc(sizeof(CPUTexture2D));
    m_backend->device_copy_to(tex_info_ptr, &dev_tex_info, sizeof(CPUTexture2D));

    mem.info.data = (uint64_t)tex_info_ptr;

    {
      thread_scoped_lock lock(texture_info_mutex);
      const uint slot = mem.slot;
      if (slot >= texture_info.size()) {
        texture_info.resize(slot + 128);
      }
      texture_info[slot] = mem.info;
      need_texture_info = true;                     // fuerza re-subida al device
    }
}


void SimpleDevice::tex_free(device_texture &mem)
{
  generic_free(mem);
  //throw std::runtime_error("Texture deallocation not supported");
}


BVHLayoutMask SimpleDevice::get_bvh_layout_mask(const uint kernel_features) const {return BVH_LAYOUT_SIMPLE;}
void SimpleDevice::const_copy_to(const char *name, void *host_ptr, const size_t size)
{
    //std::cout << "const_copy_to " << name << " of size " << size << std::endl;
    char *kg_ptr = (char *)m_backend->get_global_ptr("kernel_globals");

    // Copia al slot correspondiente (tu macro actual).
    #define KERNEL_DATA_ARRAY(t, nm) \
      if (strcmp(name, #nm) == 0) { \
        m_backend->device_copy_to(kg_ptr + offsetof(KernelParamsSimple, nm), host_ptr, size); \
        return; \
      }
    KERNEL_DATA_ARRAY(int, object_ids)
    KERNEL_DATA_ARRAY(int, prim_ids)
    KERNEL_DATA_ARRAY(int, object_sizes)
    KERNEL_DATA_ARRAY(KernelData, data)
    KERNEL_DATA_ARRAY(IntegratorStateGPU, integrator_state)
    #include "kernel/data_arrays.h"
    #undef KERNEL_DATA_ARRAY


}


void SimpleDevice::global_alloc(device_memory &mem)
{
  //check
  if (mem.is_resident(this)) {
    generic_alloc(mem);
    generic_copy_to(mem);
  }

  const_copy_to(mem.name, &mem.device_pointer, sizeof(mem.device_pointer));
}
void SimpleDevice::mem_alloc(device_memory &mem){
  //check
    std::cout << "mem_alloc " << mem.name << std::endl;
  if (mem.type == MEM_TEXTURE) {
    assert(!"mem_alloc not supported for textures.");
  }
  else if (mem.type == MEM_GLOBAL) {
    assert(!"mem_alloc not supported for global memory.");
  }
  else {
    generic_alloc(mem);
  }
}

void print_mem(SimpleDevice *device, device_memory &mem){
  int max_it = 8;
  std::cout << "printing mem " << mem.name << " (" << mem.memory_size() << " bytes)" << std::endl;
  std::cout << "host:" << std::endl;
  for(int i = 0; i < mem.memory_size(); ++i){
      std::cout << "byte " << i << ": " << (int)((char*)mem.host_pointer)[i] << std::endl;
      if(i > max_it){
          break;
      }
  }
  std::cout << "device:" << std::endl;
  void* host_ptr = malloc(mem.memory_size());
  device->m_backend->device_copy_from(host_ptr, (void*)mem.device_pointer, mem.memory_size());
  for(int i = 0; i < mem.memory_size(); ++i){
      std::cout << "byte " << i << ": " << (int)((char*)host_ptr)[i] << std::endl;
      if(i > max_it){
          break;
      }
  }
}


void SimpleDevice::global_copy_to(device_memory &mem)
{
  //check
  std::cout << "global_copy_to " << mem.name << "of size " << mem.memory_size() << std::endl;
  if (!mem.device_pointer) {
    //std::cout << "global_copy_to " << mem.name << " not allocated, allocating" << std::endl;
    generic_alloc(mem);
    generic_copy_to(mem);
  }
  else if (mem.is_resident(this)) {
   // std::cout << "global_copy_to " << mem.name << " already allocated, copying" << std::endl;
    generic_copy_to(mem);
  }

  const_copy_to(mem.name, &mem.device_pointer, sizeof(mem.device_pointer));
  //print_mem(this, mem);
}


void SimpleDevice::mem_copy_to(device_memory &mem){
  //check
  //  std::cout << "mem_copy_to " << mem.name << std::endl;
if (mem.type == MEM_GLOBAL) {
    global_copy_to(mem);
  }
  else if (mem.type == MEM_TEXTURE) {
    tex_copy_to((device_texture &)mem);
  }
  else {
    if (!mem.device_pointer) {
      generic_alloc(mem);
      generic_copy_to(mem);
    }
    else if (mem.is_resident(this)) {
      generic_copy_to(mem);
    }
  }
}

void SimpleDevice::tex_copy_to(device_texture &mem){
    if (!mem.device_pointer) {
      /* Not yet allocated on device. */
      tex_alloc(mem);
    }
    else if (!mem.is_resident(this)) {
      /* Peering with another device, may still need to create texture info and object. */
      bool texture_allocated = false;
      {
        thread_scoped_lock lock(texture_info_mutex);
        texture_allocated = mem.slot < texture_info.size() && texture_info[mem.slot].data != 0;
      }
      if (!texture_allocated) {
        tex_alloc(mem);
      }
    }
    else {
      generic_copy_to(mem);
    }
  //generic_copy_to(mem);
  //check
   //   std::cout << "texture copy error not implmeneted" << std::endl;
    //throw std::runtime_error("Texture copy not supported");
    //mem_copy_to(mem);
}


void SimpleDevice::mem_move_to_host(device_memory &mem){
  //check
  //  std::cout << "mem_move_to_host " << mem.name << std::endl;
 if (mem.type == MEM_GLOBAL) {
    global_free(mem);
    global_alloc(mem);
  }
  else if (mem.type == MEM_TEXTURE) {
    tex_free((device_texture &)mem);
    tex_alloc((device_texture &)mem);
  }
  else {
    assert(!"mem_move_to_host only supported for texture and global memory");
  }
}
void SimpleDevice::mem_zero(device_memory &mem){
    //std::cout << "mem_zero " << mem.name << std::endl;
    if (!mem.device_pointer) {
        mem_alloc(mem);
    }
    if (!mem.device_pointer) {
        return;
    }

    if (!(mem.is_shared(this) && mem.host_pointer == mem.shared_pointer)) {
        void* host_ptr = malloc(mem.memory_size());
        memset(host_ptr, 0, mem.memory_size());
        m_backend->device_copy_to((void*)mem.device_pointer, host_ptr, mem.memory_size());
        free(host_ptr);
    }
    else if (mem.host_pointer) {
        memset(mem.host_pointer, 0, mem.memory_size());
    }
}
void SimpleDevice::mem_free(device_memory &mem){
  //check
   // std::cout << "mem_free " << mem.name << std::endl;
    if (mem.type == MEM_GLOBAL) {
        global_free(mem);
    }
    else if (mem.type == MEM_TEXTURE) {
        tex_free((device_texture &)mem);
    }
    else {
        generic_free(mem);
    }
}
void SimpleDevice::mem_copy_from(device_memory &mem, const size_t y, size_t w, const size_t h, size_t elem){
  //check
   // std::cout << "mem_copy_from " << mem.name << std::endl;
  if (mem.type == MEM_TEXTURE || mem.type == MEM_GLOBAL) {
    assert(!"mem_copy_from not supported for textures.");
  }
  else if (mem.host_pointer) {
    const size_t size = elem * w * h;
    const size_t offset = elem * y * w;
    if (mem.device_pointer) {
      m_backend->device_copy_from((char *)mem.host_pointer + offset, (char *)mem.device_pointer + offset, size);
    }
    else {
      memset((char *)mem.host_pointer + offset, 0, size);
    }

    //std::cout << "trayendo de vuelta " << mem.name << " size " << size << " offset " << offset << " elem " << elem << " w " << w << " h " << h << " y " << y << std::endl;

    for(int i = 0; i < mem.memory_size(); ++i){
        //std::cout << "host byte " << i << " = " << (int)(((char*)mem.host_pointer)[i]) << std::endl;
    }
    //std::cout << std::endl;
    for(int i = 0; i < mem.memory_size(); ++i){
        //std::cout << "device byte " << i << " = " << (int)(((char*)mem.device_pointer)[i]) << std::endl;
    }
    //std::cout << std::endl;
  }
}
void SimpleDevice::get_device_memory_info(size_t &total, size_t &free){
    struct sysinfo info;
    if (sysinfo(&info) != 0) { 
        total = free = 0;
        return;
    }
    total = static_cast<size_t>(info.totalram) * info.mem_unit;
    free  = static_cast<size_t>(info.freeram) * info.mem_unit;
}


bool SimpleDevice::alloc_device(void *&device_pointer, const size_t size){
    //check
    device_pointer = m_backend->device_malloc(size);
    //std::cout << "Allocated ptr of " << size << " bytes at " << device_pointer << std::endl;
    return device_pointer != nullptr;
}
void SimpleDevice::free_device(void *device_pointer){
    //check
    if(device_pointer){
        m_backend->device_free(device_pointer);
    }
}
bool SimpleDevice::shared_alloc(void *&shared_pointer, const size_t size){
    //std::cout << "shared_alloc " << size << std::endl;
    throw std::runtime_error("NOT IMPLEMENTED");
    return false;
}
void SimpleDevice::shared_free(void *shared_pointer){
    //std::cout << "shared_free " << shared_pointer << std::endl;
    throw std::runtime_error("NOT IMPLEMENTED");
}
void *SimpleDevice::shared_to_device_pointer(const void *shared_pointer){
    //std::cout << "shared_to_device_pointer " << shared_pointer << std::endl;
    throw std::runtime_error("NOT IMPLEMENTED");
}
void SimpleDevice::copy_host_to_device(void *device_pointer, void *host_pointer, const size_t size){
  //check
    m_backend->device_copy_to(device_pointer, host_pointer, size);
}
device_ptr SimpleDevice::mem_alloc_sub_ptr(device_memory &mem, const size_t offset, size_t /*size*/)
{
  return (device_ptr)(((char *)mem.device_pointer) + mem.memory_elements_size(offset));
}

unique_ptr<DeviceQueue> SimpleDevice::gpu_queue_create() {
    return make_unique<SimpleDeviceQueue>(this);
}

void SimpleDevice::build_bvh(BVH *bvh, Progress &progress, bool refit)
{
  if (!bvh->params.top_level) return;

  std::vector<std::array<float,9>> tris;
  std::vector<int> object_ids;
  std::vector<int> object_sizes;
  std::vector<int> prim_ids;

  object_sizes.push_back(0);

  for (Object *obj : bvh->objects) {
    Geometry* geometry = obj->get_geometry();
    if (!geometry->is_mesh()){
      std::cout << "Dropping non-mesh geometry " << geometry->name << std::endl;
      object_sizes.push_back(object_sizes.back());
      continue;
    }
    Mesh* mesh = static_cast<Mesh*>(geometry);

    const Transform &M = obj->get_tfm();
    auto tp = [&](const float3 &p){
      if(!geometry->transform_applied){
        return transform_point(&M, p);
      }
      return p;
    };

    object_sizes.push_back(mesh->num_triangles() + object_sizes.back());

    for (size_t j = 0; j < mesh->num_triangles(); ++j) {
      Mesh::Triangle tri = mesh->get_triangle(j);
      
      float3 v0 = tp(mesh->get_verts()[tri.v[0]]);
      float3 v1 = tp(mesh->get_verts()[tri.v[1]]);
      float3 v2 = tp(mesh->get_verts()[tri.v[2]]);

      
      tris.push_back({v0.x,v0.y,v0.z, v1.x,v1.y,v1.z, v2.x,v2.y,v2.z});
      object_ids.push_back(obj->get_device_index()); 
      prim_ids.push_back((int)j);
    }
  }

  object_ids_mem.alloc(object_ids.size());
  prim_ids_mem.alloc(prim_ids.size());
  object_sizes_mem.alloc(object_sizes.size());

  for (size_t i = 0; i < object_ids.size(); ++i) {
    object_ids_mem[i] = object_ids[i];
    prim_ids_mem[i]   = prim_ids[i];
  }

  for (size_t i = 0; i < object_sizes.size(); ++i) {
    object_sizes_mem[i] = object_sizes[i];
  }

  object_ids_mem.copy_to_device();
  prim_ids_mem.copy_to_device();
  object_sizes_mem.copy_to_device();

  m_backend->set_tris(tris);
}

CCL_NAMESPACE_END