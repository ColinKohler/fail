"""
The following functions provide a convenient method
to print and access vectors by state or actuator name.
"""
from pydrake.common.containers import namedview
from pydrake.multibody.tree import (
  JointActuatorIndex,
  JointIndex,
  BodyIndex
)

def MakeNamedViewActuation(plant, view_name):
  names = [None] * plant.get_actuation_input_port().size()
  for ind in range(plant.num_actuators()):
    actuator = plant.get_joint_actuator(JointActuatorIndex(ind))
    assert actuator.num_inputs() == 1
    names[actuator.input_start()] = actuator.name()
  return namedview(view_name, names)

# Adapted from https://github.com/RussTedrake/manipulation/blob/master/manipulation/utils.py  # noqa
def MakeNamedViewPositions(plant, view_name, add_suffix_if_single_position=False):
  names = [None] * plant.num_positions()
  for ind in range(plant.num_joints()):
    joint = plant.get_joint(JointIndex(ind))
    if joint.num_positions() == 1 and not add_suffix_if_single_position:
      names[joint.position_start()] = joint.name()
    else:
      for i in range(joint.num_positions()):
        names[joint.position_start() + i] = f"{joint.name()}_{joint.position_suffix(i)}"
  for ind in plant.GetFloatingBaseBodies():
    body = plant.get_body(ind)
    start = body.floating_positions_start()
    for i in range(7):
      names[start + i] = f"{body.name()}_{body.floating_position_suffix(i)}"
  return namedview(view_name, names)

def MakeNamedViewVelocities(plant, view_name, add_suffix_if_single_velocity=False):
  names = [None] * plant.num_velocities()
  for ind in range(plant.num_joints()):
    joint = plant.get_joint(JointIndex(ind))
    if joint.num_velocities() == 1 and not add_suffix_if_single_velocity:
      names[joint.velocity_start()] = joint.name()
    else:
      for i in range(joint.num_velocities()):
        names[joint.velocity_start() + i] = f"{joint.name()}_{joint.velocity_suffix(i)}"
  for ind in plant.GetFloatingBaseBodies():
    body = plant.get_body(ind)
    start = body.floating_velocities_start() - plant.num_positions()
    for i in range(6):
      names[start + i] = f"{body.name()}_{body.floating_velocity_suffix(i)}"
  return namedview(view_name, names)

def MakeNamedViewForces(plant, view_name, add_suffix_if_single_position=False):
  names = [None] * plant.num_joints()
  for ind in range(plant.num_joints()):
    joint = plant.get_joint(JointIndex(ind))
    names[ind] = joint.name()
  return namedview(view_name, names)

def MakeNamedViewBodyPoses(plant, view_name):
  names = [None] * plant.num_bodies()
  for ind in range(plant.num_bodies()):
    body = plant.get_body(BodyIndex(ind))
    names[ind] = body.name()
  return namedview(view_name, names)

def MakeNamedViewState(plant, view_name):
  pview = MakeNamedViewPositions(plant, f"{view_name}_pos", True)
  vview = MakeNamedViewVelocities(plant, f"{view_name}_vel", True)
  return namedview(view_name, pview.get_fields() + vview.get_fields())
