import numpy as np
from pyquaternion import Quaternion

from yumipy import YuMiRobot
from yumipy import YuMiState
from yumipy import YuMiMotionPlanner

from autolab_core import RigidTransform

def generate_waypoints():
	# Generate a pendulum-like motion for the robot to folow
	origin = np.array([0., 0., 0.20])
	l = 0.20

	waypoints = []

	for i in range(50):
		theta = np.cos(i / 10.)

		x = origin[0] - l * np.sin(theta)
		y = origin[1]
		z = origin[2] - l * np.cos(theta)

		R = Quaternion(axis=[0., 1., 0.], radians=theta).rotation_matrix 
		t = np.array([x, y, z])

		T = RigidTransform(rotation=R, translation=t, from_frame="world")
		waypoints.append(T)

	return waypoints


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

	Y.left.goto_state(PoseCalibrationLeft)

	# Set robot to origin
	raw_input("Press enter to move to begin pose...")

	R = RigidTransform.rotation_from_quaternion([0., 0.707, 0.707, 0.])
	R_origin = RigidTransform(rotation=R, from_frame="world")
	t = np.array([0.50, 0.05, 0.20])
	t_origin = RigidTransform(translation=t, from_frame="world")
	
	T_origin = t_origin * R_origin 

	print(T_origin)
	Y.left.goto_pose(T_origin)

	waypoints = generate_waypoints()

	for w_T in waypoints:
		print(w_T)
		print(t_origin * w_T * R_origin)
		raw_input()
		Y.left.goto_pose(t_origin * w_T * R_origin)

