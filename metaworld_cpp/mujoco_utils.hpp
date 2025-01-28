#pragma once

#include <mujoco/mujoco.h>

bool touching_object(mjModel *m, mjData *d, int object_geom_id);
