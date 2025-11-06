/* SPDX-FileCopyrightText: 2011-2023 Blender Foundation
 *
 * SPDX-License-Identifier: Apache-2.0 */

 #  pragma once
 
 #  include "bvh/bvh.h"
 #  include "bvh/params.h"
 #  include "device/memory.h"
 
 CCL_NAMESPACE_BEGIN
 
 class BVHSimple : public BVH {
  public:

   BVHSimple(const BVHParams &params,
            const vector<Geometry *> &geometry,
            const vector<Object *> &objects,
            Device *in_device);
 
   ~BVHSimple() override;
 
  private:
   Device *device;
 };
 
 CCL_NAMESPACE_END
 