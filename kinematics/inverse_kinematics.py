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
        """
        Calculates the squared distance between the current end-effector position
        and the target position. It ignores orientation error for simplicity/speed.

        :param dict joints: The current joint angle configuration.
        :param str effector: The name of the end effector.
        :param np.array target: The 4x4 target transform matrix.
        :return: A scalar representing the squared position error.
        """
        self.forward_kinematics(joints=joints)

        if effector not in self.chains:
            raise ValueError("Effector does not exist. :((")

        chain_joints = self.chains[effector]
        last_joint = chain_joints[-1]

        # Get the 4x4 transform of the end-effector
        current_transform = self.transforms[last_joint]

        # --- FIX: Extract and compare only the position components (last column, rows 0-2) ---
        current_position = current_transform[:3, 3]
        target_position = target[:3, 3]

        # Calculate the error vector
        error_vector = target_position - current_position

        # Return the squared magnitude (L2 norm squared) of the position error
        return np.sum(error_vector * error_vector)

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
                print(error)
                if error < MAX_ERROR:
                    print(f"Converged in {i + 1} iterations.")
                    break

        # print(func(target), self.perception.time)
        # print("final joint angles: ", joint_angles)

        return joint_angles

    def inverse_kinematics_analytical(self, effector_name, transform):
        if effector_name not in ["LLeg", "RLeg"]:
            print("Invalid Effector, switching to iterative approach")
            return self.inverse_kinematics(effector_name, transform)

        x_pos = transform[0, 3]
        y_pos = transform[1, 3]
        z_pos = transform[2, 3]

        l_upper_leg = 0.1
        l_lower_leg = 0.1029
        l_dist = np.sqrt(x_pos**2 + y_pos**2 + (z_pos + l_upper_leg) ** 2)

        knee_angle = np.pi - np.arccos(
            (l_upper_leg**2 + l_lower_leg**2 - l_dist**2)
            / (2 * l_upper_leg * l_lower_leg)
        )

        r_01 = -np.cos(x_pos) * np.sin(z_pos)
        r_11 = np.cos(x_pos) * np.cos(z_pos)

        r_20 = -np.cos(x_pos) * np.sin(y_pos)
        r_22 = np.cos(x_pos) * np.cos(y_pos)

        hip_yaw = np.arctan2(-r_01, r_11)

        hip_roll = np.arcsin(np.sin(x_pos))

        hip_angle = np.arctan2(-r_20, r_22)

        ankle_angle = -hip_angle + knee_angle
        if effector_name == "RLeg":
            return {
                "RHipYawPitch": hip_yaw,
                "RHipRoll": hip_roll,
                "RHipPitch": hip_angle,
                "RKneePitch": knee_angle,
                "RAnklePitch": ankle_angle,
                "RAnkleRoll": 0.0,  # Keep the naming convention consistent with NAO's model
            }
        elif effector_name == "LLeg":
            return {
                "LHipYawPitch": hip_yaw,
                "LHipRoll": hip_roll,
                "LHipPitch": hip_angle,
                "LKneePitch": knee_angle,
                "LAnklePitch": ankle_angle,
                "LAnkleRoll": 0.0,  # Keep the naming convention consistent with NAO's model
            }

    def set_transforms(self, effector_name, transform):
        """solve the inverse kinematics and control joints use the results"""
        joint_angles = self.inverse_kinematics_analytical(effector_name, transform)

        default_handle_1 = [0, 0.0, 0.0]
        default_handle_2 = [0, 0.0, 0.0]

        names_list = []
        times_list = []
        keys_list = []

        chain_joints = self.chains[effector_name]

        for i, joint_name in enumerate(chain_joints):
            target_angle = joint_angles[joint_name]

            bezier_key = [target_angle, default_handle_1, default_handle_2]

            names_list.append(joint_name)
            times_list.append([1.0])
            keys_list.append([bezier_key])

        # Set the complete keyframes tuple
        self.keyframes = (names_list, times_list, keys_list)


if __name__ == "__main__":
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    T[-1, 1] = 0.05
    T[-1, 2] = -0.26
    agent.set_transforms("LLeg", T)
    agent.run()
