import apriltag
import cv2
import numpy as np
import time
import csv
# from scipy.spatial.transform import Rotation
import transforms3d

from utils import *
from kalman_filter import *
from transfoms import *
from mosaic import *
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
    camera = CameraBasler(scale=0.5)
    apriltag_detector = DetectorApriltag()
    kalman = KalmanHomography()

    H_origin = None
    d_camera_origin = None

    mosaic_shape = (2000, 2000)
    mosaic = np.zeros((2000, 2000, 3), dtype='uint8')

    prev_img = None
    H3 = None
    H1 = np.eye(3)
    H2 = np.eye(3)

    t_history = []
    R_history = []
    t_actual = []
    R_actual = []

    canvas = np.zeros((2064, 1544, 3), dtype='uint8')

    _name = "output" + '.mp4'
    _fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    _out = cv2.VideoWriter(_name, _fourcc, 15.0, (camera.h*2,camera.w))

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

    waypoints = read_waypoints("yumi/link_K_1.csv", scale=0.15)
    # waypoints = sinusoid_waypoints()
    # waypoints = custom_waypoints()

    _, t_boom = read_csv_tf("yumi/link_I_1.csv")
    R = np.array([
        [0., 0., 1.],
        [0., -1., 0.],
        [1., 0., 0.]
    ])
    for i, t in enumerate(t_boom):
        t_boom[i] = np.matmul(R, t) * 0.15 * 1000

    for i, w in enumerate(waypoints):
        waypoints[i] = t_origin * w * gripper_offset * R_origin



    Y.left.goto_pose(waypoints[0])
    raw_input("Start experiment")


    tic = time.time()


    for i, w_T in enumerate(waypoints):
        # time.sleep(0.5)
        Y.left.goto_pose(w_T)

        img = camera.capture_frame()

        results = apriltag_detector.detetct_tags(img)

        # Plot the path travelled by the marker
        if results:
            H3 = apriltag_detector.calculate_homography(results, [0, 1, 2, 3, 4, 5, 6, 7])
        else:
            H3 = None

        toc = time.time()
        kalman.predict_step(0.02)
        tic = time.time()
        if H3 is not None:
            R, t = homography_decomposition(H3, camera.K)


            if H_origin is None:
                H_origin = H3
                d_camera_origin = np.dot(np.matmul(R, t), np.array([0, 0, 1]))
                print(d_camera_origin)

            # c = np.matmul(H3, np.array([0., 0., 1.]))
            # cv2.circle(img, (int(c[0]), int(c[1])), 5, (0, 0, 255), -1)
            # print(t)

            H4 = crane_transform_given(t_boom[i], t_boom[0], camera.K, d_camera_origin)
            H4 = np.matmul(H4, np.matmul(H_origin, np.linalg.inv(H3)))

            H_src_dst = np.matmul(H4, H3)

            z = np.ndarray.flatten(H_src_dst)
            kalman.update_step(z)   

            H4 = np.matmul(np.reshape(kalman.x[:9], (3, 3)), np.linalg.inv(H3))
            
            mosaic = homography_mosaic(mosaic, img, H1, H2, H3, H4)

            H1 = H4
            H2 = H3
            cropped_mosaic = mosaic[1000-int(camera.w/2.5):1000+int(camera.w/2.5), 1000-int(camera.h/2.5):1000+int(camera.h/2.5)]
            cropped_mosaic = cv2.resize(cropped_mosaic, (img.shape[1], img.shape[0]))
            warped_img = cv2.warpPerspective(img, H4, dsize=img.shape[1::-1])

            cv2.imshow("mosaic", cv2.resize(cropped_mosaic, (cropped_mosaic.shape[1]//2, cropped_mosaic.shape[0]//2))) #cv2.resize(mosaic, (5000 // 4, 5000 // 4))
            # cv2.imshow("mosaic", cv2.resize(mosaic, (2000 // 2, 2000 // 2)))
            # cv2.imshow("warped", cv2.resize(warped_img, (img.shape[1]//2, img.shape[0]//2)))

            _out.write(cv2.hconcat([img, cropped_mosaic]))



            # c = np.matmul(H, np.array([0., 0., 1.]))
            # cv2.circle(canvas, (int(c[0]), int(c[1])), 2, (255, 255, 255), -1)

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
        cv2.imshow("Image", img)
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