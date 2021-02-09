import apriltag
import cv2
import numpy as np
import time
import csv
from scipy.spatial.transform import Rotation

from utils import *
# from kalman_filter import *
# from transfoms import *
# from mosaic import *
from DetectorApriltag import DetectorApriltag
from CameraWebcam import CameraWebcam
from CameraBasler import CameraBasler

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
from pyquaternion import Quaternion

from yumipy import YuMiRobot
from yumipy import YuMiState
from yumipy import YuMiMotionPlanner

from autolab_core import RigidTransform
from yumi.waypoints import *

def main():
    camera = CameraBasler()
    apriltag_detector = DetectorApriltag()

    prev_img = None
    H = None
    H1 = np.eye(3)
    H2 = np.eye(3)

    mosaic_shape = (1000, 1000)
    mosaic = np.zeros((1000, 1000, 3), dtype='uint8')

    t_history = []
    R_history = []
    t_actual = []
    R_actual = []

    canvas = np.zeros((2064, 1544, 3), dtype='uint8')

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
    raw_input("Goto calibration pose")

    # Y.left.goto_state(PoseCalibrationLeft)

	# Set robot to origin
    raw_input("Goto origin pose")
    np_gripper_offset = np.array([-0.0162, 0.02665, -0.0400])
    gripper_offset = RigidTransform(translation=np_gripper_offset, from_frame="world")

    R = RigidTransform.rotation_from_quaternion([0., 0.707, 0.707, 0.])
    R_origin = RigidTransform(rotation=R, from_frame="world")
    t = np.array([0.45, -0.10, 0.20])
    t_origin = RigidTransform(translation=t, from_frame="world") * gripper_offset.inverse()
	
    T_origin = t_origin * R_origin 

    T_actual = []

    Y.left.goto_pose(T_origin)
    print(T_origin)

    # waypoints = read_waypoints("yumi/link_K_1.csv", scale=0.2)
    # waypoints = sinusoid_waypoints()
    waypoints = custom_waypoints()

    for i, w in enumerate(waypoints):
        waypoints[i] = t_origin * w * gripper_offset * R_origin


    Y.left.goto_pose(waypoints[0])
    raw_input("Start experiment")


    for w_T in waypoints:
        print(w_T)
        # time.sleep(0.5)
        Y.left.goto_pose(w_T)

        img = camera.capture_frame()

        results = apriltag_detector.detetct_tags(img)

        # Plot the path travelled by the marker
        if results:
            H = apriltag_detector.calculate_homography(results, [0, 1, 2, 3, 4, 5, 6])
        else:
            H = None

        if H is not None:
            R, t = homography_decomposition(H, camera.K)

            c = np.matmul(H, np.array([0., 0., 1.]))
            cv2.circle(canvas, (int(c[0]), int(c[1])), 2, (255, 255, 255), -1)

            # t = np.matmul(np.linalg.inv(R), t)

            t_history.append(t)
            R_history.append(R)

            T_actual = Y.left.get_pose()
            T_actual = T_actual.as_frames(from_frame="world", to_frame="world")

            t_actual.append((T_actual * R_origin.inverse() * gripper_offset.inverse() * R_origin).translation)
            R_actual.append(Y.left.get_pose().rotation)

        prev_img = img
        # Show the output image after AprilTag detection
        img = apriltag_detector.draw_detections(img, results)
        print(img.shape)
        cv2.imshow("Image", cv2.resize(cv2.add(img, canvas), (1544/2, 2064/2)))
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    camera.release()
    Y.left.goto_pose(T_origin)
    
    # Write results to file
    with open('results.csv', mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['x', 'y', 'z', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'actual', 'actual', 'actual', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R'])
        for i in range(len(t_history)):
            writer.writerow(list(t_history[i]) + list(R_history[i].flatten()) + list(t_actual[i]) + list(R_actual[i].flatten()))

if __name__ == "__main__":
    main()