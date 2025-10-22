'''
In this exercise you need to know how to set joint commands.

* Tasks:
    1. set stiffness of LShoulderPitch to 0
    2. set speed of HeadYaw to 0.1

* Hint: The commands are stored in action (class Action in spark_agent.py)

'''

# add PYTHONPATH
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'software_installation'))

from spark_agent import SparkAgent


class MyAgent(SparkAgent):
    def __init__(self, turn_direction=-1):
        super().__init__()
        self.turn_direction = turn_direction

    # A fun little function I wrote to make NAO shake his head. 
    # To be used when your teammate makes a rather silly play. 
    def shake_head(self, perception):
        action = super(MyAgent, self).think(perception)
        if ( self.perception.joint["HeadYaw"] <= -1 and self.turn_direction == -1 ) or ( self.perception.joint["HeadYaw"] >= 1 and self.turn_direction == 1 ): 
            self.turn_direction = self.turn_direction * ( -1 )
        action.speed["HeadYaw"] = 0.5 * self.turn_direction
        print(self.perception.joint["HeadYaw"])
        return action

    def think(self, perception):
        action = super(MyAgent, self).think(perception)
        # YOUR CODE HERE
        action.speed["HeadYaw"] = -0.1
        action.stiffness["LShoulderPitch"] = 0
        # Output test print
        # action = self.shake_head(perception)
        return action

if '__main__' == __name__:
    agent = MyAgent()
    agent.run()
