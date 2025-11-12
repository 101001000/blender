#pragma once


#include <portableRT/portableRT.hpp>



CCL_NAMESPACE_BEGIN

ccl_device_inline bool scene_intersect_valid(const ccl_private Ray *ray)
{
  return isfinite_safe(ray->P.x) && isfinite_safe(ray->D.x) && len_squared(ray->D) != 0.0f;
}

bool terminate_ray_visibility(RaySelfPrimitives self, uint object, uint prim, const uint visibility){  
  #ifdef __VISIBILITY_FLAG__
    if ((kernel_data_fetch(objects, object).visibility & visibility) == 0) {
      return false;
    }
  #endif

    if (visibility & PATH_RAY_SHADOW_OPAQUE) {
  #ifdef __SHADOW_LINKING__
      if (intersection_skip_shadow_link(nullptr, self, object)) {
        return false;
      }
  #endif
  
      if (intersection_skip_self_shadow(self, object, prim)) {
        return false;
      }
      else {
        /* Shadow ray early termination. */
        return true;
      }
    }
    else {
      if (intersection_skip_self(self, object, prim)) {
        return false;
      }
    }
  return true;
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
    isect->type = PRIM_NONE;

    prt::Ray prt_ray;
    prt_ray.origin[0] = ray->P.x;
    prt_ray.origin[1] = ray->P.y;
    prt_ray.origin[2] = ray->P.z;
    prt_ray.direction[0] = ray->D.x;
    prt_ray.direction[1] = ray->D.y;
    prt_ray.direction[2] = ray->D.z;
    prt_ray.tmin = ray->tmin;
    prt_ray.tmax = ray->tmax;

    if(!scene_intersect_valid(ray)) {
        return false;
    }

    uint id = 0;
    uint prim_id = 0;

    do{
          
      int self_object_size = 0;
      int self_prim_id = 0;
      if(ray->self.object != OBJECT_NONE){
          self_object_size = kernel_data_fetch(object_sizes, ray->self.object);
          self_prim_id = ray->self.prim - kernel_data_fetch(object_prim_offset, ray->self.object);
      }
  
      prt_ray.self_id = self_object_size + self_prim_id;
  
      auto hit = prt::closest_hit(prt_ray);
  
      if (!hit.valid) {
          return false;
      }


      id = kernel_data_fetch(object_ids, hit.primitive_id);
      prim_id = kernel_data_fetch(prim_ids, hit.primitive_id);
      const int off = kernel_data_fetch(object_prim_offset, id);

      isect->t = hit.t;
      isect->u = hit.u;
      isect->v = hit.v;
      isect->prim = off + prim_id; // Warning
      isect->type = PRIMITIVE_TRIANGLE;
      isect->object = id;

      prt_ray.tmin = hit.t;
      prt_ray.self_id = self_object_size + prim_id;

    }while(!terminate_ray_visibility(ray->self, id, prim_id, visibility));

    return true;
    
}

bool anyhit_local_hit(uint object_, uint local_object_, uint prim_, int max_hits, RaySelfPrimitives self, uint *lcg_state, LocalIntersection *local_isect, float tmax, float u, float v, bool& out)
{
#ifdef __BVH_LOCAL__
  const int object = object_;
  if (object != local_object_) {
    /* Only intersect with matching object. */
    return false;
  }

  if(kernel_data_fetch(objects, object).primitive_type != PRIMITIVE_TRIANGLE || kernel_data_fetch(objects, local_object_).primitive_type != PRIMITIVE_TRIANGLE){
    return false;
  }


  const int prim = prim_;
  if (intersection_skip_self_local(self, prim)) {
    return false;
  }

  if (max_hits == 0) {
    /* Special case for when no hit information is requested, just report that something was hit */
    out = true;
    return true;
  }

  int hit = 0;

  if (lcg_state) {
    for (int i = min(max_hits, local_isect->num_hits) - 1; i >= 0; --i) {
      if (tmax == local_isect->hits[i].t) {
        return false;
      }
    }

    hit = local_isect->num_hits++;

    if (local_isect->num_hits > max_hits) {
      hit = lcg_step_uint(lcg_state) % local_isect->num_hits;
      if (hit >= max_hits) {
        return false;
      }
    }
  }
  else {
    if (local_isect->num_hits && tmax > local_isect->hits[0].t) {
      /* Record closest intersection only.
       * Do not terminate ray here, since there is no guarantee about distance ordering in any-hit.
       */
      return false;
    }

    local_isect->num_hits = 1;
  }

  Intersection *isect = &local_isect->hits[hit];
  isect->t = tmax;
  isect->prim = prim;
  isect->object = object;
  isect->type = PRIMITIVE_TRIANGLE;

  isect->u = u;
  isect->v = v;

  /* Record geometric normal. */
  const packed_uint3 tri_vindex = kernel_data_fetch(tri_vindex, prim);
  const float3 tri_a = kernel_data_fetch(tri_verts, tri_vindex.x);
  const float3 tri_b = kernel_data_fetch(tri_verts, tri_vindex.y);
  const float3 tri_c = kernel_data_fetch(tri_verts, tri_vindex.z);

  local_isect->Ng[hit] = normalize(cross(tri_b - tri_a, tri_c - tri_a));

  /* Continue tracing (without this the trace call would return after the first hit). */
  return false;
#endif
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
  if (local_isect) {
    local_isect->num_hits = 0; /* Initialize hit count to zero. */
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

  if(!scene_intersect_valid(ray)) {
      return false;
  }

  uint object_id = 0;
  uint prim_id = 0;
  float u;
  float v;
  float t;
  bool out = false;

  do{
        
    int self_object_size = 0;
    int self_local_prim_id = 0;
    if(ray->self.object != OBJECT_NONE){
        self_object_size = kernel_data_fetch(object_sizes, ray->self.object);
        self_local_prim_id = ray->self.prim - kernel_data_fetch(object_prim_offset, ray->self.object);
    }
    prt_ray.self_id = self_object_size + self_local_prim_id;

    auto hit = prt::closest_hit(prt_ray);

    if (!hit.valid) {
      break;
    }

    u = hit.u;
    v = hit.v;
    t = hit.t;

    object_id = kernel_data_fetch(object_ids, hit.primitive_id);
    prim_id = kernel_data_fetch(prim_ids, hit.primitive_id) + kernel_data_fetch(object_prim_offset, object_id);
    

    prt_ray.tmin = t + 0.001f;
    prt_ray.self_id = self_object_size + kernel_data_fetch(prim_ids, hit.primitive_id);

  }while(!anyhit_local_hit(object_id, local_object, prim_id, max_hits, ray->self, lcg_state, local_isect, t, u, v, out));

  return (max_hits == 0) ? out : (local_isect && local_isect->num_hits > 0);
}
#endif

#ifdef __VOLUME__
ccl_device_intersect bool scene_intersect_volume(KernelGlobals kg,
                                                 const ccl_private Ray *ray,
                                                 ccl_private Intersection *isect,
                                                 const uint visibility)
{
    throw std::runtime_error("scene_intersect_volume not implemented");
    //printf("scene_intersect_volume\n");
    return false;
}
#endif

bool anyhit_shader(int prim_, uint object_, uint visibility_, float u, float v, uint max_hits, uint& num_recorded_hits, uint& num_hits, IntegratorShadowState state, float tmax, RaySelfPrimitives self, bool& clamp_far)
{
#ifdef __SHADOW_RECORD_ALL__
  int prim = prim_;
  const uint object = object_;
#  ifdef __VISIBILITY_FLAG__
  const uint visibility = visibility_;
  if ((kernel_data_fetch(objects, object).visibility & visibility) == 0) {
    return false;
  }
#  endif

  int type = 0;
  /* Triangle. */
  type = kernel_data_fetch(objects, object).primitive_type;

  if(type != PRIMITIVE_TRIANGLE){
    return false;
  }

  if (intersection_skip_self_shadow(self, object, prim)) {
    return false;
  }

#  ifdef __SHADOW_LINKING__
  if (intersection_skip_shadow_link(nullptr, self, object)) {
    return false;
  }
#  endif

#  ifndef __TRANSPARENT_SHADOWS__
  /* No transparent shadows support compiled in, make opaque. */
  return true;
#  else

  /* If no transparent shadows, all light is blocked and we can stop immediately. */
  if (num_hits >= max_hits ||
      !(intersection_get_shader_flags(nullptr, prim, type) & SD_HAS_TRANSPARENT_SHADOW))
  {
    return true;
  }

  uint record_index = num_recorded_hits;  // índice con el valor anterior
  num_recorded_hits++;
  num_hits++;

  const uint max_record_hits = min(max_hits, INTEGRATOR_SHADOW_ISECT_SIZE);
  if (record_index >= max_record_hits) {
    /* If maximum number of hits reached, find a hit to replace. */
    float max_recorded_t = INTEGRATOR_STATE_ARRAY(state, shadow_isect, 0, t);
    uint max_recorded_hit = 0;

    for (int i = 1; i < max_record_hits; i++) {
      const float isect_t = INTEGRATOR_STATE_ARRAY(state, shadow_isect, i, t);
      if (isect_t > max_recorded_t) {
        max_recorded_t = isect_t;
        max_recorded_hit = i;
      }
    }

    if (tmax >= max_recorded_t) {
      /* Accept hit, so that OptiX won't consider any more hits beyond the distance of the
       * current hit anymore. */
      clamp_far = true;
      return false;
    }

    record_index = max_recorded_hit;
  }

  INTEGRATOR_STATE_ARRAY_WRITE(state, shadow_isect, record_index, u) = u;
  INTEGRATOR_STATE_ARRAY_WRITE(state, shadow_isect, record_index, v) = v;
  INTEGRATOR_STATE_ARRAY_WRITE(state, shadow_isect, record_index, t) = tmax;
  INTEGRATOR_STATE_ARRAY_WRITE(state, shadow_isect, record_index, prim) = prim;
  INTEGRATOR_STATE_ARRAY_WRITE(state, shadow_isect, record_index, object) = object;
  INTEGRATOR_STATE_ARRAY_WRITE(state, shadow_isect, record_index, type) = type;

  /* Continue tracing. */
  return false;
#  endif /* __TRANSPARENT_SHADOWS__ */
#endif   /* __SHADOW_RECORD_ALL__ */
}

#ifdef __SHADOW_RECORD_ALL__
ccl_device_intersect bool scene_intersect_shadow_all(KernelGlobals kg,
                                                     IntegratorShadowState state,
                                                     const ccl_private Ray *ray_,
                                                     const uint visibility,
                                                     const uint max_hits,
                                                     ccl_private uint *num_recorded_hits,
                                                     ccl_private float *throughput)
{
  
  *num_recorded_hits = 0u;
  *throughput = 1.0f;
  if (!scene_intersect_valid(ray_)) {
    return false;
  }
  
  Ray ray = *ray_;
  prt::Ray prt_ray;
  prt_ray.origin[0] = ray.P.x;
  prt_ray.origin[1] = ray.P.y;
  prt_ray.origin[2] = ray.P.z;
  prt_ray.direction[0] = ray.D.x;
  prt_ray.direction[1] = ray.D.y;
  prt_ray.direction[2] = ray.D.z;
  prt_ray.tmin = ray.tmin;
  prt_ray.tmax = ray.tmax;
  
  int self_object_size = 0;
  int self_prim_id = 0;
  if(ray.self.object != OBJECT_NONE){
      //self_object_id = kernel_data_fetch(object_ids, ray->self.object);
      self_object_size = kernel_data_fetch(object_sizes, ray.self.object);
      if(ray.self.prim != PRIM_NONE){
        self_prim_id = ray.self.prim - kernel_data_fetch(object_prim_offset, ray.self.object);
      }
  }


  prt_ray.self_id = self_object_size + self_prim_id;

  uint num_hits = 0;
  bool clamp_far = false;

  // Anyhit simulation:
  for(int i = 0; i < max_hits; i++){
   
    auto hit = prt::closest_hit(prt_ray);   
    if (hit.valid) {
      int self_object_id = kernel_data_fetch(object_ids, hit.primitive_id);
      self_object_size = kernel_data_fetch(object_sizes, self_object_id);
      self_prim_id = hit.primitive_id - kernel_data_fetch(object_prim_offset, self_object_id);
      prt_ray.self_id = self_object_size + self_prim_id;
      prt_ray.tmin = hit.t;
    } else {
      break;
    }

    bool clamp_far = false;

    const int object_id = kernel_data_fetch(object_ids, hit.primitive_id);
    const int prim_id = kernel_data_fetch(prim_ids, hit.primitive_id);
    const int off = kernel_data_fetch(object_prim_offset, object_id);
    
    // call to the anyhit "shader"
    // two kind of returns: False (ignore ray) or True (terminate ray)
    bool terminate = anyhit_shader(prim_id, object_id, visibility, hit.u, hit.v, max_hits, *num_recorded_hits, num_hits, state, hit.t, ray.self, clamp_far);
    if(terminate){
      return true;
      break;
    }
    if (clamp_far) prt_ray.tmax = hit.t;    
  }
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
