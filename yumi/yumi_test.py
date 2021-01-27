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
	Y.set_v(200)
	# Y.left.set_motion_planner()

	# PoseCalibration
	PoseCalibrationLeft = YuMiState([0, -130, 30, 0, 40, 0, 135])
	PoseCalibrationRight = YuMiState([0, -130, 30, 0, 40, 0, -135])

	# Set robot to PoseCalibration
	raw_input("Press enter to move to begin pose...")

	# Y.left.goto_state(PoseCalibrationLeft)

	# Set robot to origin
	raw_input("Press enter to move to begin pose...")

	R = RigidTransform.rotation_from_quaternion([0., 0.707, 0.707, 0.])
	R_origin = RigidTransform(rotation=R, from_frame="world")
	t = np.array([0.5, -0.10, 0.15])
	t_origin = RigidTransform(translation=t, from_frame="world")
	
	T_origin = t_origin * R_origin 

	Y.left.goto_pose(T_origin)

	waypoints = read_waypoints("crane3_3D_2p.csv", scale=0.1)

	for i, w in enumerate(waypoints):
		waypoints[i] = t_origin * w * R_origin

	print(waypoints[0])

	print("clear")
	Y.left.buffer_clear()
	print("add")
	Y.left.buffer_add_all(waypoints)
	Y.left.buffer_move()

	# for w_T in waypoints:
	# 	print(w_T)
	# 	print(t_origin * w_T * R_origin)
	# 	# raw_input()
	# 	Y.left.goto_pose(t_origin * w_T * R_origin)

	Y.left.goto_pose(T_origin)
