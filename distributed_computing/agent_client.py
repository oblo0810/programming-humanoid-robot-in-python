"""In this file you need to implement remote procedure call (RPC) client

* The agent_server.py has to be implemented first (at least one function is implemented and exported)
* Please implement functions in ClientAgent first, which should request remote call directly
* The PostHandler can be implement in the last step, it provides non-blocking functions, e.g. agent.post.execute_keyframes
 * Hints: [threading](https://docs.python.org/2/library/threading.html) may be needed for monitoring if the task is done
"""

import weakref
import Pyro5.api
import threading
import os
import sys
from autograd.numpy import identity

sys.path.append(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "joint_control")
)

from keyframes import hello


class PostHandler(object):
    """the post hander wraps function to be excuted in paralle"""

    def __init__(self, obj):
        self.proxy_name = obj.proxy_name

    def _threaded_execute_keyframes(self, keyframes):
        """non-blocking call of ClientAgent.execute_keyframes"""
        try:
            # Create a new proxy inside this thread
            proxy = Pyro5.api.Proxy(self.proxy_name)
            proxy.execute_keyframes(keyframes)
        except Exception as e:
            print(f"Error in threaded execute_keyframes: {e}")
        finally:
            if "proxy" in locals():
                proxy._pyroRelease()

    def execute_keyframes(self, keyframes):
        """non-blocking call of ClientAgent.execute_keyframes"""
        # Start the new thread, targeting our new function
        thread = threading.Thread(
            target=self._threaded_execute_keyframes, args=(keyframes,)
        )
        thread.start()

    def _threaded_set_transform(self, effector_name, transform):
        """This function will run in the new thread"""
        try:
            # 1. Create a new proxy *inside this thread*
            proxy = Pyro5.api.Proxy(self.proxy_name)
            # 2. Use the new proxy (and remember to call .tolist()!)
            proxy.set_transform(effector_name, transform.tolist())
        except Exception as e:
            print(f"Error in threaded set_transform: {e}")
        finally:
            # 3. Clean up
            if "proxy" in locals():
                proxy._pyroRelease()

    def set_transform(self, effector_name, transform):
        """non-blocking call of ClientAgent.set_transform"""
        # Start the new thread
        thread = threading.Thread(
            target=self._threaded_set_transform,
            args=(effector_name, transform),
        )
        thread.start()


class ClientAgent(object):
    """ClientAgent request RPC service from remote server"""

    # YOUR CODE HERE
    def __init__(self):
        self.proxy_name = "PYRONAME:agent.main"
        self.agent = Pyro5.api.Proxy("PYRONAME:agent.main")
        self.post = PostHandler(self)

    def get_angle(self, joint_name):
        """get sensor value of given joint"""
        # YOUR CODE HERE
        return self.agent.get_angle(joint_name=joint_name)

    def set_angle(self, joint_name, angle):
        """set target angle of joint for PID controller"""
        # YOUR CODE HERE
        self.agent.set_angle(joint_name=joint_name, angle=angle)

    def get_posture(self):
        """return current posture of robot"""
        # YOUR CODE HERE
        return self.agent.get_posture()

    def execute_keyframes(self, keyframes):
        """excute keyframes, note this function is blocking call,
        e.g. return until keyframes are executed
        """
        # YOUR CODE HERE
        self.agent.execute_keyframes(keyframes)

    def get_transform(self, name):
        """get transform with given name"""
        # YOUR CODE HERE
        return self.agent.get_transform(name)

    def set_transform(self, effector_name, transform):
        """solve the inverse kinematics and control joints use the results"""
        # YOUR CODE HERE
        self.agent.set_transform(effector_name, transform.tolist())


if __name__ == "__main__":
    agent = ClientAgent()
    # TEST CODE HERE
    # I used Pyro5 for this. Some guide recommended it, but it made this task a lot more
    # difficult. Especially the multithreading part. But now everything should work.
    #
    # To run, you will need to do:
    # 1. start the pyro5 name server using `pyro5-ns`
    # 2. in a new terminal start the agent_server
    # 3. in another new terminal start the agent_client
    print(agent.get_angle("HeadYaw"))
    print(agent.get_posture())
    print(agent.get_transform("LElbowYaw"))
    agent.execute_keyframes(hello())
    agent.post.execute_keyframes(hello())
    T = identity(4)
    T[-1, 1] = 0.05
    T[-1, 2] = -0.26
    agent.set_transform("LLeg", T)
    agent.post.set_transform("LLeg", T)

