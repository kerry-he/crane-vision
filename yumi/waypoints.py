import numpy as np
from pyquaternion import Quaternion

from autolab_core import RigidTransform

import csv

def read_waypoints(file_name, scale):
    waypoints = []

    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        top_row = True
        R = Quaternion(axis=[0., 0., 1.], radians=-np.pi/2.).rotation_matrix

        for row in csv_reader:
            if not top_row:
                t = (np.array([float(row[1]), float(row[2]), float(row[3])]) * scale)
                t = np.matmul(R, t)
                T = RigidTransform(translation=t, from_frame="world")
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

if __name__ == "__main__":
    read_waypoints("crane3_3D_2p.csv", 0.1)