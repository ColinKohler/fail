from typing import Any, ClassVar, Dict, List

import numpy as np
import rclpy
import PyKDL as kdl

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from urdf_parser.urdf import URDF
from kdl_parser import kdl_tree_from_urdf_model

from franka_streaming_driver.msg import FrankaJointState
from geometry_msgs.msg import Wrench


def kdl_to_mat(m):
    """Convert kdl matrix to numpy matrix."""
    mat = np.mat(np.zeros((m.rows(), m.columns())))
    for i in range(m.rows()):
        for j in range(m.columns()):
            mat[i, j] = m[i, j]
    return mat


def joint_list_to_kdl(q):
    """Convert joint list to kdl joint array."""
    if q is None:
        return None
    if type(q) == np.matrix and q.shape[1] == 0:
        q = q.T.tolist()[0]
    q_kdl = kdl.JntArray(len(q))
    for i, q_i in enumerate(q):
        q_kdl[i] = q_i
    return q_kdl


def create_kdl_chain(str: base_link, str: end_link, str: urdf_filename = None):
    """Create KDL chain from base link to end_link"""
    if urdf_file is not None:
        urdf = URDF.load_from_parameter_server(verbose=False)
    else:
        urdf = URDF.load_xml_file(urdf_filename, verbose=False)

    kdl_tree = kdl_tree_from_urdf_model(urdf)
    kdl_chain = kdl_tree.getChain(base_link, end_link)

    return kdl_chain


def jacobian(kdl_chain, q):
    """Get the jacobian from a kdl chain"""
    num_joints = len(q)
    jac_kdl = kdl.ChainJntToJacSolver(kdl_chain)

    j_kdl = kdl.Jacobian(num_joints)
    q_kdl = joint_list_to_kdl(q)
    jac_kdl.JntToJac(q_kdl, j_kdl)

    return kdl_to_mat(j_kdl)


class FrankaFTPublisher(Node):
    ARM_JOINT_STATE_TOPIC: ClassVar[str] = "/franka/joint_states/"
    ARM_WRENCH_STATE_TOPIC: ClassVar[str] = "/franka/wrench_state/"

    def __init__(self):
        super().__init__("/frank/ft_publisher")

        self.frequnecy = 100
        self.joint_names_prefix = "panda"
        self.franka_joint_names = tuple(
            f"{self.joint_names_prefix}_joint{i}" for i in range(1, 8)
        )

        callback_group = ReentrantCallbackGroup()
        self.create_subscription(
            FrankJointState,
            self.ARM_JOINT_STATE_TOPIC,
            self.joint_state_callback,
            10,
            callback_group=callback_group,
        )

        self.publisher = self.create_publisher(
            WrenchMessage, self.ARM_WRENCH_STATE_TOPIC, 10
        )
        self.wrench_timer = self.create_timer(
            1 / self.frequency, self.publish_wrench_state, callback_group=callback_group
        )

        self.base_link = "panda_link0"
        self.ee_link = "panda_link7"
        self.kdl_chain = create_kdl_chain()

        self.arm_tau_ext = list()

    def joint_state_callback(self, msg: FrankaJointState):
        """Arm state callback."""
        self.arm_ext_torques.append(
            np.array(msg.tau_ext[: len(self.franka_joint_names)])
        )

    def publish_wrench_state(self):
        """Publish current end effector wrench."""
        tau_ext = self.tau_ext[-1]
        jac = (kdl_chain, q)
        wrench = np.dot(tau_ext, jac)

        ee_wrench = WrenchStamped()
        ee_wrench.header.frame_id = "panda_link0"
        ee_wrench.wrench.force = wrench[:3]
        ee_wrench.wrench.torque = wrench[3:]
        self.publlisher.publish(ee_wrench)


def main(args: Any = None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(cam)

    try:
        executor.spin()
    finally:
        executor.shutdown()
        cam.destroy_node()


if __name__ == "__main__":
    main()
