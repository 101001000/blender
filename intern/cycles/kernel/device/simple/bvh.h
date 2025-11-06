#pragma once


#include <portableRT/portableRT.hpp>



CCL_NAMESPACE_BEGIN

ccl_device_inline bool scene_intersect_valid(const ccl_private Ray *ray)
{
  return isfinite_safe(ray->P.x) && isfinite_safe(ray->D.x) && len_squared(ray->D) != 0.0f;
}

ccl_device_intersect bool scene_intersect(KernelGlobals kg,
    const ccl_private Ray *ray,
    const uint visibility,
    ccl_private Intersection *isect)
{

    isect->t = ray->tmax;
    isect->u = 0.0f;
    isect->v = 0.0f;
    isect->prim = PRIM_NONE;
    isect->object = OBJECT_NONE;
    isect->type = PRIMITIVE_NONE;


    if(!scene_intersect_valid(ray)) {
        return false;
    }

    prt::Ray prt_ray;
    prt_ray.origin[0] = ray->P.x;
    prt_ray.origin[1] = ray->P.y;
    prt_ray.origin[2] = ray->P.z;
    prt_ray.direction[0] = ray->D.x;
    prt_ray.direction[1] = ray->D.y;
    prt_ray.direction[2] = ray->D.z;
    prt_ray.tmin = ray->tmin;
    prt_ray.tmax = ray->tmax;
    prt_ray.self_id = ray->self.prim;
    auto hit = prt::closest_hit(prt_ray);


    if (!hit.valid) {
        return false;
    }

    const int id = kernel_data_fetch(object_ids, hit.primitive_id);

    isect->t = hit.t;
    isect->u = hit.u;
    isect->v = hit.v;
    isect->prim = hit.primitive_id;
    isect->type = PRIMITIVE_TRIANGLE;
    isect->object = id;

    return true;
    
}


#ifdef __BVH_LOCAL__
template<bool single_hit = false>
ccl_device_intersect bool scene_intersect_local(KernelGlobals kg,
                                                const ccl_private Ray *ray,
                                                ccl_private LocalIntersection *local_isect,
                                                const int local_object,
                                                ccl_private uint *lcg_state,
                                                const int max_hits)
{
    //printf("scene_intersect_local\n");
    return false;
}
#endif

#ifdef __VOLUME__
ccl_device_intersect bool scene_intersect_volume(KernelGlobals kg,
                                                 const ccl_private Ray *ray,
                                                 ccl_private Intersection *isect,
                                                 const uint visibility)
{
    //printf("scene_intersect_volume\n");
    return false;
}
#endif

#ifdef __SHADOW_RECORD_ALL__
ccl_device_intersect bool scene_intersect_shadow_all(KernelGlobals kg,
                                                     IntegratorShadowState state,
                                                     const ccl_private Ray *ray,
                                                     const uint visibility,
                                                     const uint max_hits,
                                                     ccl_private uint *num_recorded_hits,
                                                     ccl_private float *throughput)
{
    //printf("scene_intersect_shadow_all\n");
    return false;
}
#endif

ccl_device_intersect bool scene_intersect_shadow(KernelGlobals kg,
    const ccl_private Ray *ray,
    const uint visibility)
{
    Intersection isect;
    return scene_intersect(kg, ray, visibility, &isect);
}


CCL_NAMESPACE_END
