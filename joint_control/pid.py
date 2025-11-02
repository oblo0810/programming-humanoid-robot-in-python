"""In this exercise you need to implement the PID controller for joints of robot.

* Task:
    1. complete the control function in PIDController with prediction
    2. adjust PID parameters for NAO in simulation

* Hints:
    1. the motor in simulation can simple modelled by angle(t) = angle(t-1) + speed * dt
    2. use self.y to buffer model prediction
"""

# add PYTHONPATH
import os
import sys

sys.path.append(
    os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "..", "software_installation"
    )
)

import numpy as np
from collections import deque
from spark_agent import SparkAgent, JOINT_CMD_NAMES


class PIDController(object):
    """a discretized PID controller, it controls an array of servos,
    e.g. input is an array and output is also an array
    """

    def __init__(self, dt, size):
        """
        @param dt: step time
        @param size: number of control values
        @param delay: delay in number of steps
        """
        self.dt = dt
        self.u = np.zeros(size)
        self.e1 = np.zeros(size)
        self.e2 = np.zeros(size)
        self.size = size
        # ADJUST PARAMETERS BELOW
        delay = 0  # The delay is optional but in a real model you need it for good performance.
        self.Kp = 20
        self.Ki = 0.1
        self.Kd = 0.1
        self.delay_steps = delay
        # I changed this a little bit since I had issues with y. I believe the old implementation saved scalars in place of vectors leading to slight errors.
        self.y = deque(maxlen=delay + 1)
        for _ in range(self.delay_steps + 1):
            self.y.append(np.zeros(self.size))

    def set_delay(self, delay):
        """
        @param delay: delay in number of steps
        """
        self.delay_steps = int(delay)

        # Create a new, clean deque
        self.y = deque(maxlen=self.delay_steps + 1)
        for _ in range(self.delay_steps + 1):
            self.y.append(np.zeros(self.size))

        # RESET THE CONTROLLER'S MEMORY
        # This is the step you were missing.
        self.u.fill(0)
        self.e1.fill(0)
        self.e2.fill(0)

    def control(self, target, sensor):
        """apply PID control
        @param target: reference values
        @param sensor: current values from sensor
        @return control signal
        """
        # calc u:
        feedback = self.y[-1] + (sensor - self.y[0])
        e = target - feedback

        c1 = self.Kp + self.Ki * self.dt + self.Kd / self.dt
        c2 = self.Kp + 2 * self.Kd / self.dt
        c3 = self.Kd / self.dt

        self.u = self.u + c1 * e - c2 * self.e1 + c3 * self.e2

        # update parameters
        self.e2 = self.e1
        self.e1 = e

        # calculate new y and update model
        y_tilde_new = self.y[-1] + self.u * self.dt
        self.y.append(y_tilde_new)

        return self.u


class PIDAgent(SparkAgent):
    def __init__(
        self,
        simspark_ip="localhost",
        simspark_port=3100,
        teamname="DAInamite",
        player_id=0,
        sync_mode=True,
    ):
        super(PIDAgent, self).__init__(
            simspark_ip, simspark_port, teamname, player_id, sync_mode
        )
        self.joint_names = JOINT_CMD_NAMES.keys()
        number_of_joints = len(self.joint_names)
        self.joint_controller = PIDController(dt=0.01, size=number_of_joints)
        self.target_joints = {k: 0 for k in self.joint_names}

    def think(self, perception):
        action = super(PIDAgent, self).think(perception)
        """calculate control vector (speeds) from
        perception.joint:   current joints' positions (dict: joint_id -> position (current))
        self.target_joints: target positions (dict: joint_id -> position (target)) """
        joint_angles = np.asarray(
            [perception.joint[joint_id] for joint_id in JOINT_CMD_NAMES]
        )
        target_angles = np.asarray(
            [
                self.target_joints.get(joint_id, perception.joint[joint_id])
                for joint_id in JOINT_CMD_NAMES
            ]
        )
        u = self.joint_controller.control(target_angles, joint_angles)
        action.speed = dict(zip(JOINT_CMD_NAMES.keys(), u))  # dict: joint_id -> speed
        return action


if __name__ == "__main__":
    agent = PIDAgent()
    agent.target_joints["HeadYaw"] = 1.0
    agent.run()
