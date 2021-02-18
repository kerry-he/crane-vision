import numpy as np
from pyquaternion import Quaternion

from autolab_core import RigidTransform

import csv

def read_csv_tf(file_name):
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
    waypoints = []

    R_list, t_list = read_csv_tf(file_name)

    R_90_z = Quaternion(axis=[0., 0., 1.], radians=np.pi/2.).rotation_matrix
    R_90_z_T = RigidTransform(rotation=R_90_z, from_frame="world")

    t_R = np.array([
        [0., -1., 0.],
        [0., 0., 1.],
        [-1., 0., 0.]
    ])
    t_R_T = RigidTransform(rotation=t_R, from_frame="world")

    offset_K = np.array([
        [0., 1., 0.],
        [-1., 0., 0.],
        [0., 0., 1.]
    ])

    for i in range(len(R_list)):
        rotation = Quaternion(axis=[1., 0., 0.], angle=np.pi/12).rotation_matrix
        R = np.matmul(np.matmul(np.linalg.inv(t_R), np.matmul(R_list[i], np.linalg.inv(offset_K))), t_R)#np.eye(3)#rotation#R_list[i]
        t = t_list[i] * scale
        t = np.matmul(np.linalg.inv(t_R), t)
        print(t)
        T = RigidTransform(rotation=R, translation=t, from_frame="world")
        # T = R_90_z_T * T
        waypoints.append(T)

        top_row = False

    offset = waypoints[0].translation
    offset_T = RigidTransform(translation=-offset, from_frame="world")

    for i, w in enumerate(waypoints):
        waypoints[i] = offset_T * w    

    return waypoints

def transform_waypoints(waypoints, T):
    for i, w in enumerate(waypoints):
        waypoints[i] = T * w

    return waypoints

def sinusoid_waypoints():
	# Generate a pendulum-like motion for the robot to follow
	origin = np.array([0., 0., 0.20])
	l = 0.20

	waypoints = []

	for i in range(150):
		theta = np.cos(i / 10.)/2
		origin[0] = -np.sin(np.cos(i / 20.)/2) * 0.3

		x = origin[0] - l * np.sin(theta)
		y = origin[1]
		z = origin[2] - l * np.cos(theta)

		R = Quaternion(axis=[0., 1., 0.], radians=theta).rotation_matrix 
		t = np.array([x, y, z])

		T = RigidTransform(rotation=R, translation=t, from_frame="world")
		waypoints.append(T)

	return waypoints

def custom_waypoints():
    # Generate a pendulum-like motion for the robot to follow
    origin = np.array([0., 0., 0.20])

    waypoints = []

    t_K = np.array([
        [0., 1., 0.],
        [-1., 0., 0.],
        [0., 0., 1.]
    ])

    for i in range(150):
        theta = np.sin(i / 10.)
        R = Quaternion(axis=[0., 0., 1.], radians=theta/4.).rotation_matrix
        T = RigidTransform(rotation=R, from_frame="world")
        waypoints.append(T)

    return waypoints

if __name__ == "__main__":
    waypoints = read_waypoints("link_K_1.csv", 0.1)
    print(waypoints)