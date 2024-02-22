import os
import copy
import numpy as np
import numpy.random as npr
import gymnasium as gym
from pathlib import Path
from functools import partial
import pydot
from spatialmath.base import q2r
from collections import deque
from scipy.signal import butter, filtfilt

from pydrake.math import (
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
)
from pydrake.geometry import (
    Box,
    MeshcatVisualizer,
    ClippingRange,
    ColorRenderCamera,
    DepthRange,
    DepthRenderCamera,
    MakeRenderEngineVtk,
    RenderCameraCore,
    RenderEngineVtkParams,
    RenderLabel,
)
from pydrake.multibody.parsing import Parser
from pydrake.multibody.tree import (
    FixedOffsetFrame,
    LinearSpringDamper,
    SpatialInertia,
    UnitInertia,
)
from pydrake.multibody.plant import (
    AddMultibodyPlant,
    ExternallyAppliedSpatialForce_,
    MultibodyPlant,
    MultibodyPlantConfig,
    CoulombFriction,
)
from pydrake.multibody.math import SpatialForce
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.meshcat import ContactVisualizer
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import (
    DiagramBuilder,
    EventStatus,
    LeafSystem,
    PublishEvent,
)
from pydrake.systems.primitives import ConstantVectorSource, Multiplexer, PassThrough
from pydrake.systems.controllers import (
    JointStiffnessController,
    InverseDynamicsController,
)
from pydrake.systems.sensors import (
    CameraInfo,
    RgbdSensor,
    ImageDepth32F,
    PixelType,
    Image,
)
from pydrake.solvers import Solve
from pydrake.common import FindResourceOrThrow
from pydrake.common.value import AbstractValue

from fail.env.drake_env.base_env import BaseEnv
from fail.env.drake_env.named_view_helpers import (
    MakeNamedViewBodyPoses,
    MakeNamedViewForces,
)

from bdai.optimization.trajectory_optimization.stations import get_station
from bdai.optimization.trajectory_optimization.trajectory_optimization import (
    TrajectoryOptimization,
)

SIM_TIME_STEP = 0.001  # 1kHz
GYM_TIME_STEP = 0.1  # 10Hz
CONTROLLER_TIME_STEP = 0.001  # 1kHz
STATE_TIME_STEP = 0.01
GYM_TIME_LIMIT = 10
CONTACT_MODEL = "hydroelastic_with_fallback"
CONTACT_SOLVER = "sap"  #'tamsi'
HOME_Q = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, np.pi / 4])


class ObservationPublisher(LeafSystem):
    def __init__(self, pose_view, joint_view):
        super().__init__()

        self.pose_view = pose_view
        self.joint_view = joint_view
        self.state_len = int(GYM_TIME_STEP / STATE_TIME_STEP)

        self.poses_port = self.DeclareAbstractInputPort(
            "poses", AbstractValue.Make([RigidTransform()])
        )
        self.force_port = self.DeclareAbstractInputPort(
            "forces", AbstractValue.Make([SpatialForce()])
        )

        self.DeclareVectorOutputPort("goal_observation", 3, self.getGoalObs)
        self.DeclareVectorOutputPort(
            "state_observation", 9 * self.state_len, self.getStateObs
        )

        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=STATE_TIME_STEP,
            offset_sec=STATE_TIME_STEP,
            update=self.stateUpdate,
        )

        self.inital_force = None
        self.state_history = list()

    def stateUpdate(self, context, discrete_state):
        poses = self.pose_view(self.poses_port.Eval(context))
        hand_force = self.joint_view(self.force_port.Eval(context)).panda_hand_joint
        hand_T_world = poses.panda_hand.inverse()
        world_force = hand_force.Rotate(hand_T_world.rotation())

        force = world_force.get_coeffs()
        if self.initial_force is None:
            self.initial_force = copy.copy(force)
        force -= self.initial_force

        robot_T_world = poses.panda_link0.inverse()
        world_T_peg = poses.peg_base_link
        world_T_hole = poses.peg_hole_base_link

        robot_T_peg = robot_T_world.multiply(world_T_peg)
        robot_T_hole = robot_T_world.multiply(world_T_hole)

        self.force_history.append(force)
        self.pose_history.append(robot_T_peg.translation())

    def getGoalObs(self, context, output):
        poses = self.pose_view(self.poses_port.Eval(context))
        robot_T_world = poses.panda_link0.inverse()
        world_T_hole = poses.peg_hole_base_link
        robot_T_hole = robot_T_world.multiply(world_T_hole)

        output.set_value(robot_T_hole.translation())

    def getStateObs(self, context, output):
        poses = self.pose_view(self.poses_port.Eval(context))
        robot_T_world = poses.panda_link0.inverse()
        world_T_peg = poses.peg_base_link
        world_T_hole = poses.peg_hole_base_link

        robot_T_peg = robot_T_world.multiply(world_T_peg)
        robot_T_hole = robot_T_world.multiply(world_T_hole)

        time = context.get_time()
        if time == 0:
            force = np.array([[0] * 6] * self.state_len)
            pose = np.array([robot_T_peg.translation()] * self.state_len)

            self.initial_force = None
            self.force_history = [[0] * 6] * self.state_len
            self.pose_history = pose.tolist()
        else:
            pose = np.array(self.pose_history[-self.state_len :])
            force = np.array(self.force_history[-self.state_len :])

        fs = 100  # Sample rate (Hz)
        cutoff = 10 / (0.5 * fs)
        b, a = butter(1, cutoff, btype="low", analog=False)
        force = np.array([filtfilt(b, a, force[:, i]) for i in range(6)]).transpose(
            1, 0
        )

        obs = np.concatenate((pose, force), axis=1).reshape(-1)
        output.set_value(obs)


class ActionSystem(LeafSystem):
    def __init__(self, trajopt):
        super().__init__()

        self.trajopt = trajopt

        self.action_port = self.DeclareVectorInputPort("dxy_actions", 16)
        self.panda_state_port = self.DeclareVectorInputPort("panda_states", 18)
        self.DeclareVectorOutputPort("ik_input", 18, self.getIKAction)

    def getIKAction(self, context, output):
        # Get delta actions
        robot_T_hand = self.action_port.Eval(context).reshape((4, 4))
        panda_state = self.panda_state_port.Eval(context)[:7]

        self.trajopt.set_state(q=panda_state)
        q_sol = None
        ik_attempt = 0
        while q_sol is None and ik_attempt < 10:
            q_sol = self.trajopt.solve_ik(
                target_pose=robot_T_hand, q_init=panda_state, collision_checking=False
            )
            if q_sol is not None:
                q_sol = None if np.any(np.abs(q_sol - panda_state) > 0.2) else q_sol
            ik_attempt += 1
        if q_sol is None:
            q_sol = np.zeros(7)

        # Solve ik
        output.set_value(np.concatenate((q_sol, [0.0] * 11)))


class RewardSystem(LeafSystem):
    def __init__(self, pose_view):
        super().__init__()

        self.pose_view = pose_view
        self.input_port = self.DeclareAbstractInputPort(
            "poses", AbstractValue.Make([RigidTransform()])
        )
        self.DeclareVectorOutputPort("reward", 1, self.getReward)

    def getReward(self, context, output):
        poses = self.pose_view(self.EvalAbstractInput(context, 0).get_value())

        robot_T_world = poses.panda_link0.inverse()
        world_T_peg = poses.peg_insertion_link
        world_T_hole = poses.peg_hole_base_link

        robot_T_peg = robot_T_world.multiply(world_T_peg)
        robot_T_hole = robot_T_world.multiply(world_T_hole)

        peg_xyz = copy.copy(robot_T_peg.translation())
        peg_xyz[2] -= 0.1  # TODO: Define this offset as the peg insertion height
        hole_xyz = copy.copy(robot_T_hole.translation())

        reward = np.allclose(hole_xyz, peg_xyz, atol=5e-3)

        output[0] = reward


class PegInsertionEnv(object):
    def __init__(self, meshcat=None, time_limit=5, debug=False):
        self.time_limit = time_limit

        # Setup plant, builder, and scene
        self.builder = DiagramBuilder()
        multibody_plant_config = MultibodyPlantConfig(
            time_step=SIM_TIME_STEP,
            contact_model=CONTACT_MODEL,
            discrete_contact_solver=CONTACT_SOLVER,
        )
        self.plant, self.scene_graph = AddMultibodyPlant(
            multibody_plant_config, self.builder
        )
        self.scene_graph.AddRenderer(
            "renderer", MakeRenderEngineVtk(RenderEngineVtkParams())
        )
        self.trajopt = TrajectoryOptimization(debug=False)

        # Init drake scene
        self.initScene()

        # Make NamedViews
        self.pose_view = MakeNamedViewBodyPoses(self.plant, "Poses")
        self.joint_view = MakeNamedViewForces(self.plant, "Joints")

        # Setup observation, action, and reward systems
        self.initSystems()

        # Start simulator
        if meshcat:
            MeshcatVisualizer.AddToBuilder(self.builder, self.scene_graph, meshcat)

        self.diagram = self.builder.Build()
        self.simulator = Simulator(self.diagram)
        self.simulator.Initialize()
        self.simulator.set_monitor(self.monitor)

        # Save Drake graph
        if debug:
            graph = pydot.graph_from_dot_data(
                self.diagram.GetGraphvizString(max_depth=2)
            )[0]
            graph.write_png("test.png")

    def initScene(self, debug=False):
        parser = Parser(self.plant)
        parser.SetAutoRenaming(True)

        # Add assets to the plant
        ws_path = Path(os.environ["BDAI"]) / "projects" / "_experimental" / "drake_ws"
        self.panda_file = FindResourceOrThrow(
            "drake/manipulation/models/franka_description/urdf/panda_arm_hand.urdf"
        )
        self.panda = parser.AddModels(self.panda_file)[0]
        peg = parser.AddModels(
            str(ws_path / "models/peg_insertion/round_peg/Peg.urdf")
        )[0]
        hole_id = parser.AddModels(
            str(ws_path / "models/peg_insertion/round_peg/fixture/Hole.urdf")
        )[0]

        # Build manipulator station setup
        robot, static_scene_shapes = get_station("cherry")
        self.world_T_robot = RigidTransform(pose=robot["world_pose"])
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetFrameByName("panda_link0"),
            self.world_T_robot,
        )

        for name, shape in static_scene_shapes.items():
            pose = shape["world_pose"]
            size = shape["size"]
            color = np.array([0.5, 0.5, 0.5, 1.0])

            object_shape = Box(size)
            world_T_object = RigidTransform(pose=pose)
            object_instance = self.plant.AddModelInstance(name)
            object_inertia = SpatialInertia(
                mass=1.0, p_PScm_E=np.zeros(3), G_SP_E=UnitInertia(1.0, 1.0, 1.0)
            )
            object_body = self.plant.AddRigidBody(name, object_instance, object_inertia)
            self.plant.WeldFrames(
                self.plant.world_frame(), object_body.body_frame(), world_T_object
            )
            self.plant.RegisterVisualGeometry(
                object_body, RigidTransform(), object_shape, name + "_vis", color
            )
            self.plant.RegisterCollisionGeometry(
                object_body,
                RigidTransform(),
                object_shape,
                name + "_col",
                CoulombFriction(0.9, 0.8),
            )

        # Setup sensors
        intrinsics = CameraInfo(
            width=640,
            height=480,
            fov_y=np.pi / 6,
        )
        core = RenderCameraCore(
            "renderer", intrinsics, ClippingRange(0.01, 10.0), RigidTransform()
        )
        color_cam = ColorRenderCamera(core, show_window=False)
        depth_cam = DepthRenderCamera(core, DepthRange(0.01, 10.0))

        world_T_sensor = RigidTransform(
            RollPitchYaw([(7 / 8) * -np.pi, 0, np.pi / 2]), p=[0.5, 0.45, 1.65]
        )
        self.rgbd_sensor = RgbdSensor(
            self.plant.GetBodyFrameIdOrThrow(self.plant.world_body().index()),
            X_PB=world_T_sensor,
            color_camera=color_cam,
            depth_camera=depth_cam,
        )

        self.builder.AddSystem(self.rgbd_sensor)
        self.builder.Connect(
            self.scene_graph.get_query_output_port(),
            self.rgbd_sensor.query_object_input_port(),
        )

        # Fix peg hole on table into robot base frame
        hole_base = self.plant.AddFrame(
            frame=FixedOffsetFrame(
                name="peg_hole_base",
                P=self.plant.GetFrameByName("panda_link0"),
                X_PF=RigidTransform(),
                model_instance=None,
            )
        )
        self.plant.WeldFrames(
            hole_base, self.plant.GetFrameByName("peg_hole_base_link"), RigidTransform()
        )

        # Fix peg into gripper frame
        self.plant.WeldFrames(
            self.plant.GetFrameByName("panda_hand"),
            self.plant.GetFrameByName("peg_base_link"),
            RigidTransform(RollPitchYaw([np.pi / 2, 0, 0]), p=[0, 0, 0.1]),
        )

        self.plant.Finalize()
        self.plant.set_name("plant")

    def initSystems(self):
        # Create plant containing only the robot for the JointStiffnessController to operate on
        self.controller_plant = MultibodyPlant(time_step=CONTROLLER_TIME_STEP)
        cparser = Parser(self.controller_plant)
        cparser.AddModels(self.panda_file)[0]
        self.controller_plant.WeldFrames(
            self.controller_plant.world_frame(),
            self.controller_plant.GetFrameByName("panda_link0"),
            self.world_T_robot,
        )
        self.controller_plant.Finalize()

        # Create robot controller using controller plant
        N = self.controller_plant.num_positions()
        controller = self.builder.AddNamedSystem(
            "panda_controller",
            JointStiffnessController(
                plant=self.controller_plant,
                kp=np.array([60, 60, 60, 60, 25, 15, 5, 5, 5]) * 10,
                # kd=[5, 5, 5, 5, 3, 2.5, 1.5, 1.5, 1.5]
                kd=[100] * 9,
            ),
        )

        # Action system
        action = self.builder.AddSystem(ActionSystem(self.trajopt))
        self.builder.ExportInput(action.GetInputPort("dxy_actions"), "actions")
        self.builder.Connect(
            self.plant.get_state_output_port(self.panda),
            action.GetInputPort("panda_states"),
        )
        self.builder.Connect(
            self.plant.get_state_output_port(self.panda),
            controller.get_input_port_estimated_state(),
        )
        self.builder.Connect(
            action.get_output_port(), controller.get_input_port_desired_state()
        )
        self.builder.Connect(
            controller.get_output_port(),
            self.plant.get_actuation_input_port(self.panda),
        )

        # Observation system
        obs_pub = self.builder.AddSystem(
            ObservationPublisher(self.pose_view, self.joint_view)
        )
        self.builder.Connect(
            self.plant.GetOutputPort("body_poses"), obs_pub.GetInputPort("poses")
        )
        self.builder.Connect(
            self.plant.get_reaction_forces_output_port(), obs_pub.GetInputPort("forces")
        )
        self.builder.ExportOutput(
            obs_pub.GetOutputPort("state_observation"), "state_observation"
        )
        self.builder.ExportOutput(
            obs_pub.GetOutputPort("goal_observation"), "goal_observation"
        )
        # self.builder.ExportOutput(self.rgbd_sensor.GetOutputPort('depth_image_32f'), 'depth_observations')

        # Reward system
        reward = self.builder.AddSystem(RewardSystem(self.pose_view))
        self.builder.Connect(
            self.plant.GetOutputPort("body_poses"), reward.get_input_port(0)
        )
        self.builder.ExportOutput(reward.get_output_port(), "reward")

    def reset(self, diagram_context, seed):
        npr.seed(seed)
        plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, diagram_context
        )

        q_sol = None
        while q_sol is None:
            # Randomly sample poses for the hole and the initial gripper pose
            hole_pose = RigidTransform(
                RollPitchYaw([np.pi / 2, 0, 0]),
                p=[npr.uniform(0.45, 0.55), npr.uniform(-0.05, 0.05), 0.0],
            )

            hole_frame = self.plant.GetFrameByName("peg_hole_base")
            hole_frame.SetPoseInParentFrame(plant_context, X_PF=hole_pose)

            robot_T_hand = RigidTransform(
                RotationMatrix(q2r([0, 1, 0, 0])),
                copy.copy(hole_pose.translation())
                + npr.uniform([-0.05, -0.05, 0.275], [0.05, 0.05, 0.35]),
            ).GetAsMatrix4()

            # Solve IK and set panda to joint jositions
            self.trajopt.set_state(q=HOME_Q)
            q_sol = None
            ik_attempt = 0
            while q_sol is None and ik_attempt < 10:
                q_sol = self.trajopt.solve_ik(
                    target_pose=robot_T_hand, q_init=HOME_Q, collision_checking=False
                )
                ik_attempt += 1

        self.plant.SetPositions(
            plant_context,
            self.plant.GetModelInstanceByName("panda"),
            np.concatenate((q_sol, [0.025] * 2)),  # Add gripper joints onto IK solution
        )

        self.diagram.ForcedPublish(diagram_context)

    def monitor(self, context):
        """Monitors the simulation for termination conditions."""
        plant_context = self.plant.GetMyContextFromRoot(context)
        poses = self.pose_view(
            self.plant.GetOutputPort("body_poses").Eval(plant_context)
        )

        # Episode timed out
        if context.get_time() > self.time_limit:
            return EventStatus.ReachedTermination(self.diagram, "time limit")

        # Peg inserted into hole
        robot_T_world = poses.panda_link0.inverse()
        world_T_peg = poses.peg_insertion_link
        world_T_hole = poses.peg_hole_base_link

        robot_T_peg = robot_T_world.multiply(world_T_peg)
        robot_T_hole = robot_T_world.multiply(world_T_hole)

        peg_xyz = copy.copy(robot_T_peg.translation())
        peg_xyz[2] -= 0.1
        hole_xyz = copy.copy(robot_T_hole.translation())

        in_bounds = [
            (peg_xyz[0] > 0.3 and peg_xyz[0] < 0.6),
            (peg_xyz[1] > -0.15 and peg_xyz[1] < 0.15),
            (peg_xyz[2] > 0.0 and peg_xyz[2] < 0.36),
        ]
        if not np.all(in_bounds):
            return EventStatus.ReachedTermination(
                self.diagram, "Gripper outside workspace."
            )

        if np.allclose(hole_xyz, peg_xyz, atol=5e-3):
            return EventStatus.ReachedTermination(
                self.diagram, "Peg inserted into hole."
            )

        return EventStatus.Succeeded()


def createPegInsertionEnv(meshcat=None, time_limit=GYM_TIME_LIMIT):
    drake_sim = PegInsertionEnv(meshcat=meshcat, time_limit=time_limit)

    # Define action space: (dx, dy)
    na = 3
    low_a = [-0.02] * na
    high_a = [0.02] * na
    action_space = gym.spaces.Box(
        low=np.asarray(low_a, dtype="float32"),
        high=np.asarray(high_a, dtype="float32"),
        dtype=np.float32,
    )

    # Define observation space: Peg position (x, y, z) + Peg Hole (x, y, z) + Peg F/T (Fx, Fy, Mx, My)
    pose_low = [0.4, -0.1, 0.0, 0.4, -0.1, 0.0] * 128
    pose_high = [0.6, 0.1, 0.35, 0.6, 0.1, 0.35] * 128
    force_low = [-10.0] * 6 * 128
    force_high = [10.0] * 6 * 128
    observation_spaces = [
        gym.spaces.Box(
            low=np.asarray(pose_low), high=np.asarray(pose_high), dtype=np.float32
        ),
        gym.spaces.Box(
            low=np.asarray(force_low), high=np.asarray(force_high), dtype=np.float32
        ),
    ]

    env = BaseEnv(
        drake_sim=drake_sim,
        time_step=GYM_TIME_STEP,
        action_space=action_space,
        observation_spaces=observation_spaces,
        reward="reward",
        action_port_id="actions",
        observation_port_ids=["state_observation"],
    )

    return env
