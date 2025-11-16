"""In this file you need to implement remote procedure call (RPC) server

* There are different RPC libraries for python, such as xmlrpclib, json-rpc. You are free to choose.
* The following functions have to be implemented and exported:
 * get_angle
 * set_angle
 * get_posture
 * execute_keyframes
 * get_transform
 * set_transform
* You can test RPC server with ipython before implementing agent_client.py
"""

# add PYTHONPATH
import os
import sys
import Pyro5.api
import threading
import autograd.numpy as np

sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "kinematics")
)

from inverse_kinematics import InverseKinematicsAgent


@Pyro5.api.expose
class ServerAgent(InverseKinematicsAgent):
    """ServerAgent provides RPC service"""

    def get_angle(self, joint_name):
        """get sensor value of given joint"""
        if joint_name in self.perception.joint:
            return self.perception.joint[joint_name]
        else:
            raise KeyError(f"{joint_name} does not exist. :(")

    def set_angle(self, joint_name, angle):
        """set target angle of joint for PID controller"""
        if joint_name in self.target_joints:
            self.target_joints[joint_name] = angle
            print(f"Locked and set {joint_name} to {angle}")
            print(self.target_joints)
        else:
            # Still raise the error, the lock will be released automatically
            raise KeyError(f"{joint_name} does not exist. :(")

    def get_posture(self):
        """return current posture of robot"""
        # YOUR CODE HERE
        return self.recognize_posture(self.perception)

    def execute_keyframes(self, keyframes):
        """excute keyframes, note this function is blocking call,
        e.g. return until keyframes are executed
        """
        # YOUR CODE HERE
        self.keyframes = keyframes

    def get_transform(self, name):
        """get transform with given name"""
        # YOUR CODE HERE
        if name in self.transforms:
            return self.transforms[name].tolist()
        else:
            raise KeyError(f"{name} does not exist. ;(")

    def set_transform(self, effector_name, transform):
        """solve the inverse kinematics and control joints use the results"""
        # YOUR CODE HERE
        if effector_name in self.chains:
            transform = np.array(transform)
            self.set_transforms(effector_name, transform)
        else:
            raise KeyError(f"{effector_name} does not exist. ;;(((")


if __name__ == "__main__":
    # BEFORE RUNNING YOU NEED TO RUN A PYRO5 NAME SERVER USING:
    # pyro5-ns
    # before you will need to install pyro5 using pip install Pyro5
    deamon = Pyro5.server.Daemon()
    name_server = Pyro5.api.locate_ns()
    agent = ServerAgent()
    uri = deamon.register(agent)
    name_server.register("agent.main", uri)
    print(f"Server is running. URI: {uri}")
    # run the Pyro deamon in a seperate thread to avoid blocking the agent.run
    deamon_thread = threading.Thread(target=deamon.requestLoop, daemon=True)
    deamon_thread.start()
    agent.run()
