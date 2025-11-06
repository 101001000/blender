/* SPDX-FileCopyrightText: 2011-2023 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */


 #  include "bvh/simple.h"
 
 #  include "scene/mesh.h"
 #  include "scene/object.h"
 
 #  include "device/simple/device_impl.h"
 
 CCL_NAMESPACE_BEGIN
 
 BVHSimple::BVHSimple(const BVHParams &params_,
                    const vector<Geometry *> &geometry,
                    const vector<Object *> &objects,
                    Device *in_device)
     : BVH(params_, geometry, objects),
       device(in_device)
 {
  params.bvh_layout = BVH_LAYOUT_SIMPLE;
 }
 
 BVHSimple::~BVHSimple()
 {
 }
 
 CCL_NAMESPACE_END
 
 