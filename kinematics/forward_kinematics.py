"""In this exercise you need to implement forward kinematics for NAO robot

* Tasks:
    1. complete the kinematics chain definition (self.chains in class ForwardKinematicsAgent)
       The documentation from Aldebaran is here:
       http://doc.aldebaran.com/2-1/family/robots/bodyparts.html#effector-chain
    2. implement the calculation of local transformation for one joint in function
       ForwardKinematicsAgent.local_trans. The necessary documentation are:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    3. complete function ForwardKinematicsAgent.forward_kinematics, save the transforms of all body parts in torso
       coordinate into self.transforms of class ForwardKinematicsAgent

* Hints:
    1. the local_trans has to consider different joint axes and link parameters for different joints
    2. Please use radians and meters as unit.
"""

# add PYTHONPATH
import os
import sys

sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "joint_control")
)

import numpy as np

from numpy.matlib import identity
from math import cos, sin

from recognize_posture import PostureRecognitionAgent


class ForwardKinematicsAgent(PostureRecognitionAgent):
    def __init__(
        self,
        simspark_ip="localhost",
        simspark_port=3100,
        teamname="DAInamite",
        player_id=0,
        sync_mode=True,
    ):
        super(ForwardKinematicsAgent, self).__init__(
            simspark_ip, simspark_port, teamname, player_id, sync_mode
        )
        self.transforms = {n: identity(4) for n in self.joint_names}

        # chains defines the name of chain and joints of the chain
        self.chains = {
            "Head": ["HeadYaw", "HeadPitch"],
            "LArm": [
                "LShoulderPitch",
                "LShoulderRoll",
                "LElbowYaw",
                "LElbowRoll",
                # "LWristYaw", does not seem to exist
                # "LHand", actuator not a joint, so I will leave it out for now
            ],
            "LLeg": [
                "LHipYawPitch",
                "LHipRoll",
                "LHipPitch",
                "LKneePitch",
                "LAnklePitch",
                "LAnkleRoll",
            ],
            "RLeg": [
                "RHipYawPitch",
                "RHipRoll",
                "RHipPitch",
                "RKneePitch",
                "RAnklePitch",
                "RAnkleRoll",
            ],
            "RArm": [
                "RShoulderPitch",
                "RShoulderRoll",
                "RElbowYaw",
                "RElbowRoll",
                # "RWristYaw", does not seem to exist
                # "RHand", actuator not a joint, so you know the deal
            ],
        }

        self.joint_params = {
            "HeadYaw": {"axis": "z", "x": 0, "y": 0, "z": 0.12650},
            "HeadPitch": {"axis": "y", "x": 0, "y": 0, "z": 0},
            "LShoulderPitch": {
                "axis": "y",
                "x": 0,
                "y": 0.098,
                "z": 0.100,
            },
            "LShoulderRoll": {"axis": "z", "x": 0, "y": 0, "z": 0},
            "LElbowYaw": {
                "axis": "x",
                "x": 0.105,
                "y": 0.015,
                "z": 0,
            },
            "LElbowRoll": {"axis": "z", "x": 0, "y": 0, "z": 0},
            "LHipYawPitch": {
                "axis": "z",
                "x": 0,
                "y": 0.050,
                "z": -0.085,
            },
            "LHipRoll": {"axis": "x", "x": 0, "y": 0, "z": 0},
            "LHipPitch": {"axis": "y", "x": 0, "y": 0, "z": 0},
            "LKneePitch": {
                "axis": "y",
                "x": 0,
                "y": 0,
                "z": -0.100,
            },
            "LAnklePitch": {
                "axis": "y",
                "x": 0,
                "y": 0,
                "z": -0.10290,
            },
            "LAnkleRoll": {"axis": "x", "x": 0, "y": 0, "z": 0},
            "RAnkleRoll": {"axis": "x", "x": 0, "y": 0, "z": 0},
            "RShoulderPitch": {
                "axis": "y",
                "x": 0,
                "y": 0.098,
                "z": 0.100,
            },
            "RShoulderRoll": {"axis": "z", "x": 0, "y": 0, "z": 0},
            "RElbowYaw": {
                "axis": "x",
                "x": 0.105,
                "y": 0.015,
                "z": 0,
            },
            "RElbowRoll": {"axis": "z", "x": 0, "y": 0, "z": 0},
            "RHipYawPitch": {
                "axis": "z",
                "x": 0,
                "y": 0.050,
                "z": -0.085,
            },
            "RHipRoll": {"axis": "x", "x": 0, "y": 0, "z": 0},
            "RHipPitch": {"axis": "y", "x": 0, "y": 0, "z": 0},
            "RKneePitch": {
                "axis": "y",
                "x": 0,
                "y": 0,
                "z": -0.100,
            },
            "RAnklePitch": {
                "axis": "y",
                "x": 0,
                "y": 0,
                "z": -0.10290,
            },
        }

    def think(self, perception):
        self.forward_kinematics(perception.joint)
        return super(ForwardKinematicsAgent, self).think(perception)

    def local_trans(self, joint_name, joint_angle):
        """calculate local transformation of one joint

        :param str joint_name: the name of joint
        :param float joint_angle: the angle of joint in radians
        :return: transformation
        :rtype: 4x4 matrix
        """
        T = identity(4)
        # YOUR CODE HERE
        joint_params = self.joint_params[joint_name]

        T_link = np.array(
            [
                [1, 0, 0, joint_params["x"]],
                [0, 1, 0, joint_params["y"]],
                [0, 0, 1, joint_params["z"]],
                [0, 0, 0, 1],
            ]
        )

        R = identity(4)
        if joint_params["axis"] == "x":
            R = np.array(
                [
                    [1, 0, 0, 0],
                    [0, cos(joint_angle), -sin((joint_angle)), 0],
                    [0, sin(joint_angle), cos(joint_angle), 0],
                    [0, 0, 0, 1],
                ]
            )
        if joint_params["axis"] == "y":
            R = np.array(
                [
                    [cos(joint_angle), 0, sin(joint_angle), 0],
                    [0, 1, 0, 0],
                    [-sin(joint_angle), 0, cos(joint_angle), 0],
                    [0, 0, 0, 1],
                ]
            )
        if joint_params["axis"] == "z":
            R = np.array(
                [
                    [cos(joint_angle), sin(joint_angle), 0, 0],
                    [-sin(joint_angle), cos(joint_angle), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ]
            )

        T = T_link @ R

        return T

    def forward_kinematics(self, joints):
        """forward kinematics

        :param joints: {joint_name: joint_angle}
        """
        for chain_joints in self.chains.values():
            T = identity(4)
            for joint in chain_joints:
                angle = joints[joint]
                Tl = self.local_trans(joint, angle)
                # YOUR CODE HERE
                T = T @ Tl
                self.transforms[joint] = T
                print(self.transforms)


if __name__ == "__main__":
    agent = ForwardKinematicsAgent()
    agent.forward_kinematics(agent.perception.joint)
    agent.run()
