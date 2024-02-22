from typing import Callable, Optional, Union

import numpy as np
import numpy.random as npr
import gym
import collections
from spatialmath.base import q2r

from pydrake.common import RandomGenerator
from pydrake.systems.analysis import Simulator, SimulatorStatus
from pydrake.systems.framework import (
    Context,
    InputPort,
    InputPortIndex,
    OutputPortIndex,
    PortDataType,
    System,
)


def getOutputPort(system, id):
    """Convience wrapper to get the correct output port"""
    if isinstance(id, OutputPortIndex):
        return system.get_output_port(id)
    return system.GetOutputPort(id)


class BaseEnv(gym.Env):
    def __init__(
        self,
        drake_sim,
        time_step: float,
        action_space: gym.spaces.Space,
        observation_space: gym.spaces.Space,
        reward: Union[Callable[[System, Context], float], OutputPortIndex, str],
        action_port_id: Union[InputPort, InputPortIndex, str],
        observation_port_ids: list[str],
    ):
        super().__init__()

        self.drake_sim = drake_sim

        assert time_step > 0
        self.time_step = time_step

        self.action_space = action_space
        self.observation_space = observation_space

        if isinstance(reward, (OutputPortIndex, str)):
            self.reward_port_id = reward
            self.reward = None
        elif callable(reward):
            self.reward_port_id = None
            self.reward = reward
        else:
            raise ValueError("Invalid reward argument.")

        self.action_port_id = action_port_id
        self.observation_port_ids = observation_port_ids

        self.generator = RandomGenerator()

        self._setup()

    def _setup(self):
        """Completes the simulation setup"""
        system = self.drake_sim.simulator.get_system()

        # Setup action port
        if isinstance(self.action_port_id, InputPortIndex):
            self.action_port = system.get_input_port(self.action_port_id)
        else:
            self.action_port = system.GetInputPort(self.action_port_id)

        # Setup observation port
        self.observation_ports = [
            getOutputPort(system, id) for id in self.observation_port_ids
        ]
        self.depth_port = getOutputPort(system, "depth_observation")
        self.rgb_port = getOutputPort(system, "rgb_observation")

        # Setup reward
        if self.reward_port_id:
            reward_port = getOutputPort(system, self.reward_port_id)
            self.reward = lambda system, context: reward_port.Eval(context)[0]

    def step(self, d_action):
        assert self.drake_sim.simulator, "Must first call reset()"

        context = self.drake_sim.simulator.get_context()
        plant_context = self.drake_sim.plant.GetMyContextFromRoot(context)
        time = context.get_time()

        d_action = np.clip(d_action, self.action_space.low, self.action_space.high)
        # TODO: Probably need to clip this to ws bounds
        poses = self.drake_sim.pose_view(
            self.drake_sim.plant.GetOutputPort("body_poses").Eval(plant_context)
        )

        robot_T_world = poses.panda_link0.inverse()
        world_T_hand = poses.panda_hand
        robot_T_hand = robot_T_world.multiply(world_T_hand).GetAsMatrix4()
        robot_T_hand[:3, :3] = q2r([0, 1, 0, 0])
        robot_T_hand[:2, 3] = np.clip(
            robot_T_hand[:2, 3] + np.array([d_action[0], d_action[1]]),
            np.array([0.3, -0.15]),
            np.array([0.6, 0.15]),
        )

        self.action_port.FixValue(context, robot_T_hand.reshape(-1))
        timeout = False

        status = self.drake_sim.simulator.AdvanceTo(time + self.time_step)
        obs = self.get_obs(context)
        reward = self.reward(self.drake_sim.simulator.get_system(), context)
        terminated = not timeout and (
            status.reason() == SimulatorStatus.ReturnReason.kReachedTerminationCondition
        )

        return obs, reward, terminated, dict()

    def reset(self):
        context = self.drake_sim.simulator.get_mutable_context()
        context.SetTime(0)
        self.drake_sim.simulator.Initialize()
        self.drake_sim.simulator.get_system().SetDefaultContext(context)
        self.drake_sim.reset(context)

        return self.get_obs(context)

    def get_obs(self, context):
        obs = collections.OrderedDict()
        for port in self.observation_ports:
            obs[port.get_name()] = port.Eval(context)
        return obs

    def render(self, mode):
        assert self.drake_sim.simulator, "Must first call reset()"

        if mode == "human":
            self.drake_sim.simulator.get_system().ForcedPublish(
                self.drake_sim.simulator.get_context()
            )
        elif mode == "rgb_array":
            rgb = self.rgb_port.Eval(self.drake_sim.simulator.get_context()).data[
                :, :, :3
            ]
            return rgb
        else:
            raise Exception("Invalid render mode specified.")
        return

    def seed(self, seed=None):
        self._seed = seed
        self.drake_sim._seed = seed
        self.np_random = npr.default_rng(seed)
