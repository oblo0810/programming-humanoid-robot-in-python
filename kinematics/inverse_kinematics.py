"""In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
"""

from forward_kinematics import ForwardKinematicsAgent
from numpy.matlib import identity
import numpy as np
from autograd import grad

MAX_ITERATIONS = 1000
LEARNING_RATE = 1e-2
MAX_ERROR = 1e-4


class InverseKinematicsAgent(ForwardKinematicsAgent):
    """
    Calculates the distance between the current joints and the target.
    """

    def error_func(self, joints, effector, target):
        self.forward_kinematics(joints=joints)
        if effector not in self.chains:
            raise ValueError("Effector does not exist. :((")

        chain_joints = self.chains[effector]
        last_joint = chain_joints[-1]
        current_transform = self.transforms[last_joint]
        error_matrix = target - current_transform

        return np.sum(error_matrix * error_matrix)

    def inverse_kinematics(self, effector_name, transform):
        """solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        """
        joint_angles = []
        # calculate inverse kinematics using Jacobian method
        target = transform
        chain_joints = self.chains[effector_name]

        def gradient(t):
            return self.error_func(agent.perception.joint, effector_name, t)

        joint_angles_as_list = np.array(
            [self.perception.joint[name] for name in chain_joints]
        )

        func_grad = grad(gradient)

        for i in range(MAX_ITERATIONS):
            e = gradient(transform)
            d = func_grad(transform)

        # print(func(target), self.perception.time)

        return joint_angles

    def set_transforms(self, effector_name, transform):
        """solve the inverse kinematics and control joints use the results"""
        # YOUR CODE HERE
        self.inverse_kinematics(effector_name, transform)
        self.keyframes = ([], [], [])  # the result joint angles have to fill in


if __name__ == "__main__":
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    T[-1, 1] = 0.05
    T[-1, 2] = -0.26
    agent.set_transforms("LLeg", T)
    agent.run()
