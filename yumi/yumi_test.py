import numpy as np
from pyquaternion import Quaternion

from yumipy import YuMiRobot
from yumipy import YuMiState
from yumipy import YuMiMotionPlanner

from autolab_core import RigidTransform
from waypoints import *

if __name__ == '__main__':
	# YuMi Instantiation
	print("Instantiate YuMi")
	Y = YuMiRobot()#(arm_type='remote')
	# planner = YuMiMotionPlanner()

	# Set speed of both arms
	Y.set_v(100)
	# Y.left.set_motion_planner()

	# PoseCalibration
	PoseCalibrationLeft = YuMiState([0, -130, 30, 0, 40, 0, 135])
	PoseCalibrationRight = YuMiState([0, -130, 30, 0, 40, 0, -135])

	PoseOrigin = YuMiState([-128.18, -73.52, 41.74, 115.34, 137.16, 6.38, 135.0])

	# Set robot to PoseCalibration
	raw_input("Press enter to move to begin pose...")

	# Y.left.goto_state(PoseCalibrationLeft)	
	# Y.left.goto_state(PoseOrigin)

	# Set robot to origin

	raw_input("Press enter to move to begin pose...")
	

	np_gripper_offset = np.array([-0.016, 0.0, -0.040])
	gripper_offset = RigidTransform(translation=np_gripper_offset, from_frame="world")

	R = RigidTransform.rotation_from_quaternion([0., 0.707, 0.707, 0.])
	R_origin = RigidTransform(rotation=R, from_frame="world")
	t = np.array([0.45, -0.10, 0.20])
	t_origin = RigidTransform(translation=t, from_frame="world") * gripper_offset.inverse()
	
	T_origin = t_origin * R_origin 

	Y.left.goto_pose(T_origin)

	print(Y.left.get_state())

	# waypoints = read_waypoints("link_K_1.csv", scale=0.2)
	waypoints = sinusoid_waypoints()
	# waypoints = custom_waypoints()

	

	for i, w in enumerate(waypoints):
		waypoints[i] = t_origin * w * gripper_offset * R_origin

	# print(waypoints[0])

	# print("clear")
	# Y.left.buffer_clear()
	# print("add")
	# Y.left.buffer_add_all(waypoints)
	# Y.left.buffer_move()

	for w_T in waypoints:
		# print(w_T)
		# print(t_origin * w_T * R_origin)
		# raw_input("Next waypoint?")
		Y.left.goto_pose(w_T)

	Y.left.goto_pose(T_origin)
