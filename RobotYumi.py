##################################################################
# ROBOT YUMI
# Class to help initialise and use the yumipy library
##################################################################

# Libraries
import numpy as np

# yumipy imports
from yumipy import YuMiRobot
from yumipy import YuMiState
from yumipy import YuMiMotionPlanner
from autolab_core import RigidTransform

# Own libraries
from read_waypoints import *

class RobotYumi():
    def __init__(self):

        # YuMi Instantiation
        print("Instantiate YuMi")
        self.robot = YuMiRobot()
        
        # Set speed of both arms
        self.robot.set_v(200)

        # PoseCalibration
        pose_zero = YuMiState([0, -130, 30, 0, 40, 0, 135])
        pose_recalibrate = YuMiState(
            [-128.18, -73.52, 41.74, 115.34, 137.16, 6.38, 135.0])

        # Set robot to a calibration pose
        raw_input("Goto calibration pose")
        # self.robot.left.goto_state(pose_zero)
        # self.robot.left.goto_state(pose_recalibrate)


        # Set robot to origin
        # Define offset between gripper and camera origin
        gripper_offset = np.array([-0.0162, 0.02665, -0.0400])
        self.gripper_offset = RigidTransform(
            translation=gripper_offset, from_frame="world")

        # Define an origin for waypoints to be defined about
        R_origin = RigidTransform.rotation_from_quaternion(
            [0., 0.707, 0.707, 0.])
        t_origin = np.array([0.45, -0.10, 0.20])

        self.R_origin = RigidTransform(rotation=R_origin, from_frame="world")
        self.t_origin = RigidTransform(
            translation=t_origin, from_frame="world") * self.gripper_offset.inverse()

        self.T_origin = self.t_origin * self.R_origin

        raw_input("Goto origin pose")
        self.robot.left.goto_pose(self.T_origin)


        # Read waypoints
        scale = 0.15
        self.waypoints = read_waypoints("data/waypoints/link_K_1.csv", scale=scale)
        # self.waypoints = pendulum_waypoints()
        # self.waypoints = rotational_waypoints()

        for i, w in enumerate(self.waypoints):
            self.waypoints[i] = self.t_origin * w * self.gripper_offset * self.R_origin

        _, self.t_boom = read_csv_tf("data/waypoints/link_K_1.csv")
        R = np.array([
            [0., 0., 1.],
            [0., -1., 0.],
            [1., 0., 0.]
        ])
        # Rotate boom coordinates to be in same frame as hook block
        for i, t in enumerate(self.t_boom):
            self.t_boom[i] = np.matmul(R, t) * scale * 1000

        self.i = 0

        self.robot.left.goto_pose(self.waypoints[0])
        raw_input("Start experiment")


    def execute_waypoints(self):
        for w_T in waypoints:
            self.robot.left.goto_pose(w_T)


    def step_waypoint(self):
        self.robot.left.goto_pose(self.waypoints[self.i])

        self.i += 1
        return self.i - 1


    def go_home(self):
        self.robot.left.goto_pose(self.T_origin)


if __name__ == "__main__":
    yumi = RobotYumi()
    yumi.execute_waypoints()
    yumi.go_home()
