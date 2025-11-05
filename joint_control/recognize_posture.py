"""In this exercise you need to use the learned classifier to recognize current posture of robot

* Tasks:
    1. load learned classifier in `PostureRecognitionAgent.__init__`
    2. recognize current posture in `PostureRecognitionAgent.recognize_posture`

* Hints:
    Let the robot execute different keyframes, and recognize these postures.

"""

from angle_interpolation import AngleInterpolationAgent
from keyframes import hello
import pickle

ROBOT_POSE_CLF = "robot_pose.pkl"


class PostureRecognitionAgent(AngleInterpolationAgent):
    def __init__(
        self,
        simspark_ip="localhost",
        simspark_port=3100,
        teamname="DAInamite",
        player_id=0,
        sync_mode=True,
    ):
        super(PostureRecognitionAgent, self).__init__(
            simspark_ip, simspark_port, teamname, player_id, sync_mode
        )
        self.posture = "unknown"
        with open(ROBOT_POSE_CLF, "rb") as f:
            self.posture_classifier = pickle.load(f)

    def think(self, perception):
        self.posture = self.recognize_posture(perception)
        return super(PostureRecognitionAgent, self).think(perception)

    def recognize_posture(self, perception):
        posture = "unknown"
        # Recognize posture
        keys = [
            "LHipYawPitch",
            "LHipRoll",
            "LHipPitch",
            "LKneePitch",
            "RHipYawPitch",
            "RHipRoll",
            "RHipPitch",
            "RKneePitch",
            "AngleX",
            "AngleY",
        ]
        joint_values = [perception.joint[key] for key in keys[:-2]]
        joint_values.extend(perception.imu)
        prediction_value = self.posture_classifier.predict([joint_values])[0]
        print(prediction_value)
        return posture


if __name__ == "__main__":
    agent = PostureRecognitionAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
