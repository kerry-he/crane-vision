##################################################################
# WAYPOINTS
# Functions to read wayopint CSV files, or generate custom paths.
##################################################################

# Libraries
import numpy as np
import csv
from pyquaternion import Quaternion

# yumipy imports
from autolab_core import RigidTransform

def read_csv_tf(file_name):
    # Reads input CSV file and stores in R and t arrays
    R_list = []
    t_list = []

    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            R = np.array([
                [float(row[0]), float(row[1]), float(row[2])],
                [float(row[3]), float(row[4]), float(row[5])],
                [float(row[6]), float(row[7]), float(row[8])]
            ])

            t = np.array([float(row[9]), float(row[10]), float(row[11])])

            R_list.append(R)
            t_list.append(t)
    
    return R_list, t_list

def read_waypoints(file_name, scale):
    # Generate RigidTransform waypoints and transform them to the correct frame from CSV file

    waypoints = []

    R_list, t_list = read_csv_tf(file_name)

    # Define rotation matrices of gripper and K coordinate frames
    R_gripper = np.array([
        [0., -1., 0.],
        [0., 0., 1.],
        [-1., 0., 0.]
    ])

    R_K = np.array([
        [0., 1., 0.],
        [-1., 0., 0.],
        [0., 0., 1.]
    ])

    for i in range(len(R_list)):
        R = np.matmul(np.matmul(np.linalg.inv(R_gripper), np.matmul(R_list[i], np.linalg.inv(R_K))), R_gripper)
        t = t_list[i] * scale
        t = np.matmul(np.linalg.inv(R_gripper), t)
        T = RigidTransform(rotation=R, translation=t, from_frame="world")
        waypoints.append(T)

    offset = -waypoints[0].translation
    offset_T = RigidTransform(translation=offset, from_frame="world")

    # Set initial waypoint as origin
    for i, w in enumerate(waypoints):
        waypoints[i] = offset_T * w    

    return waypoints

def pendulum_waypoints():
	# Generate a pendulum-like motion for the robot to follow
	origin = np.array([0., 0., 0.20])
	l = 0.20

	waypoints = []

	for i in range(150):
		theta = np.cos(i / 10.)/2
        # Add in translational component to pivot
		origin[0] = -np.sin(np.cos(i / 20.)/2) * 0.3

        # Calculate swinging displacement and orientation
		x = origin[0] - l * np.sin(theta)
		y = origin[1]
		z = origin[2] - l * np.cos(theta)

		R = Quaternion(axis=[0., 1., 0.], radians=theta).rotation_matrix 
		t = np.array([x, y, z])

		T = RigidTransform(rotation=R, translation=t, from_frame="world")
		waypoints.append(T)

	return waypoints

def rotational_waypoints():
    # Generate waypoints which rotates about a specified axis
    origin = np.array([0., 0., 0.20])

    waypoints = []

    for i in range(150):
        theta = np.sin(i / 10.)
        R = Quaternion(axis=[0., 0., 1.], radians=theta/4.).rotation_matrix
        T = RigidTransform(rotation=R, from_frame="world")
        waypoints.append(T)

    return waypoints