#include <mujoco/mujoco.h>

bool touching_object(mjModel *m, mjData *d, int object_geom_id) {
  // TODO: We can probably extract the IDs only once
  int leftpad_geom_id = mj_name2id(m, mjOBJ_GEOM, "leftpad_geom");
  int rightpad_geom_id = mj_name2id(m, mjOBJ_GEOM, "rightpad_geom");

  mjtNum leftpad_object_contact_force = 0.0;
  mjtNum rightpad_object_contact_force = 0.0;

  // Loop through all contacts
  for (int i = 0; i < d->ncon; i++) {
    mjContact* contact = d->contact + i;

    // Check left pad contacts
    if ((contact->geom1 == leftpad_geom_id &&
         contact->geom2 == object_geom_id) ||
        (contact->geom1 == object_geom_id &&
         contact->geom2 == leftpad_geom_id)) {
      leftpad_object_contact_force += d->efc_force[contact->efc_address];
    }

    // Check right pad contacts
    if ((contact->geom1 == rightpad_geom_id &&
         contact->geom2 == object_geom_id) ||
        (contact->geom1 == object_geom_id &&
         contact->geom2 == rightpad_geom_id)) {
      rightpad_object_contact_force += d->efc_force[contact->efc_address];
    }
  }

  return (leftpad_object_contact_force > 0.0 &&
          rightpad_object_contact_force > 0.0);
}
