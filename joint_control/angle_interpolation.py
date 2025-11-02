"""In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
"""

from pid import PIDAgent
from keyframes import hello


class AngleInterpolationAgent(PIDAgent):
    def __init__(
        self,
        simspark_ip="localhost",
        simspark_port=3100,
        teamname="DAInamite",
        player_id=0,
        sync_mode=True,
    ):
        super(AngleInterpolationAgent, self).__init__(
            simspark_ip, simspark_port, teamname, player_id, sync_mode
        )
        self.keyframes = ([], [], [])
        self.start_time = None

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        target_joints["RHipYawPitch"] = target_joints[
            "LHipYawPitch"
        ]  # copy missing joint in keyframes
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def find_current_segment_id(self, times, current_time):
        """
        finds the start index of a segment
        """
        # Die Logik in angle_interpolation stellt bereits sicher,
        # dass wir nicht vor dem ersten oder nach dem letzten Keyframe sind.
        for j in range(len(times) - 1):
            if times[j] <= current_time < times[j + 1]:
                return j

        # Fallback, sollte eigentlich nicht erreicht werden, aber zur Sicherheit:
        return len(times) - 2

    def angle_interpolation(self, keyframes, perception):
        # Set animation start time if it has not been set yet
        if self.start_time is None:
            self.start_time = perception.time

        names, times, keys = keyframes
        target_joints = perception.joint.copy()
        # YOUR CODE HERE
        # print(perception.time)
        motion_time = perception.time - self.start_time
        for i in range(len(names)):
            joint_name = names[i]
            joint_times = times[i]
            joint_keys = keys[i]
            if joint_name not in perception.joint:
                # Needed because some joints in the keyframes dont exist.
                continue

            # print(joint_name)
            # print(joint_times)
            # print(joint_keys)

            target_angle = 0.0

            # Case 1: before the first keyframe:
            if motion_time < joint_times[0]:
                target_angle = joint_keys[0][0]

            # Case 2: after the last keyframe:
            elif motion_time >= joint_times[-1]:
                target_angle = joint_keys[-1][0]

            # Case 3: between first and last keyframe
            else:
                segment_id = self.find_current_segment_id(joint_times, motion_time)
                start_time = joint_times[segment_id]
                end_time = joint_times[segment_id + 1]
                start_key = joint_keys[segment_id]
                end_key = joint_keys[segment_id + 1]
                segment_duration = end_time - start_time

                # stores at which point of the segment we are
                t = (motion_time - start_time) / segment_duration

                P0 = start_key[0]
                P3 = end_key[0]
                P1 = P0 + start_key[2][2]
                P2 = P3 + end_key[1][2]

                c0 = (1 - t) ** 3
                c1 = 3 * ((1 - t) ** 2) * t
                c2 = 3 * (1 - t) * t**2
                c3 = t**3

                Bi = c0 * P0 + c1 * P1 + c2 * P2 + c3 * P3
                target_angle = Bi

            target_joints[joint_name] = target_angle

        return target_joints


if __name__ == "__main__":
    agent = AngleInterpolationAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
