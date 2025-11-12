"""In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
"""

from typing import final
from forward_kinematics import ForwardKinematicsAgent
import autograd.numpy as np
from autograd.numpy import identity
from autograd import grad

MAX_ITERATIONS = 1000
LEARNING_RATE = 0.1
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
        # q holds our joint angles for this joint. These are the ones we want to adjust
        joint_angles = np.array([self.perception.joint[name] for name in chain_joints])

        def func_to_minimize(joint_angles_array):
            current_joints = self.perception.joint.copy()

            # We update the current joints in our array
            for i, joint_name in enumerate(chain_joints):
                current_joints[joint_name] = joint_angles_array[i]

            error = self.error_func(current_joints, effector_name, target)
            # print("\n\n\n\n", error)
            return error

        func_grad = grad(func_to_minimize)

        for i in range(MAX_ITERATIONS):
            d = func_grad(joint_angles)
            joint_angles = joint_angles - d * LEARNING_RATE

            # We only check for error every 50 runs
            if i % 50 == 0 or i == MAX_ITERATIONS - 1:
                error = func_to_minimize(joint_angles)
                # print(error)
                if error < MAX_ERROR:
                    print(f"Converged in {i + 1} iterations.")
                    break

        # print(func(target), self.perception.time)
        # print("final joint angles: ", joint_angles)

        return joint_angles

    def set_transforms(self, effector_name, transform):
        """solve the inverse kinematics and control joints use the results"""
        joint_angles = self.inverse_kinematics(effector_name, transform)

        default_handle_1 = [0, 0.0, 0.0]
        default_handle_2 = [0, 0.0, 0.0]

        names_list = []
        times_list = []
        keys_list = []

        chain_joints = self.chains[effector_name]

        for i, joint_name in enumerate(chain_joints):
            target_angle = joint_angles[i]

            bezier_key = [target_angle, default_handle_1, default_handle_2]

            names_list.append(joint_name)
            times_list.append([1.0])
            keys_list.append([bezier_key])

        # Set the complete keyframes tuple
        self.keyframes = (names_list, times_list, keys_list)

        # Optional: Print to verify the new structure
        # print("Generated Keyframes:")
        # print("Names:", self.keyframes[0])
        # print("Times:", self.keyframes[1])
        # print("Keys:", self.keyframes[2])


if __name__ == "__main__":
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    T[-1, 1] = 0.05
    T[-1, 2] = -0.26
    agent.set_transforms("LLeg", T)
    agent.run()
