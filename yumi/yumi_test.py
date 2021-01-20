import numpy as np

from yumipy import YuMiRobot
from yumipy import YuMiState

from autolab_core import RigidTransform

if __name__ == '__main__':
	# YuMi Instantiation
	print("Instantiate YuMi")
	Y = YuMiRobot()


	print(Y.left.get_pose())
	print(Y.left.get_state())

	# Set speed of both arms
	Y.set_v(50)
	

	# PoseCalibration
	PoseCalibrationLeft = YuMiState([0, -130, 30, 0, 40, 0, 135])
	PoseCalibrationRight = YuMiState([0, -130, 30, 0, 40, 0, -135])

	# Set robot to PoseCalibration
	raw_input("Press enter to move to begin pose...")

	left_pose = Y.left.get_pose()

	T = RigidTransform(rotation=np.eye(3), translation=np.array([0.1, 0., 0.]), from_frame="world")
	target_pose = T * left_pose

	print(target_pose)

	Y.left.goto_pose(target_pose)



	left_pose = Y.left.get_pose()

	T = RigidTransform(rotation=np.eye(3), translation=np.array([-0.2, 0., 0.]), from_frame="world")
	target_pose = T * left_pose

	print(target_pose)

	Y.left.goto_pose(target_pose)
	
	# # Y.right.goto_state(PoseCalibrationRight)
	
	# left_pose = Y.left.get_pose()

	# print(left_pose.translation)
	# print(left_pose.rotation)

	# T = RigidTransform(rotation=left_pose.rotation, translation=left_pose.translation + np.array([0.2, 0.1, 0.1]))

	# Y.left.goto_pose(T)


	# Y.left.goto_pose(T)

	# left_pose = Y.left.get_pose()

	# print(left_pose.translation)
	# print(left_pose.rotation)

