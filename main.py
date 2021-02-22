##################################################################
# MAIN
# Main script to run the algorithm. Can be ran alongside YuMi
# waypoints to sync images and waypoints together.
##################################################################

# Libraries
import cv2
import numpy as np
import time
import csv
import transforms3d

# Own libraries
import transfoms
import process_pose
from utils import scale_img, homography_decomposition

# Own classes
from DetectorApriltag import DetectorApriltag
from Camera import CameraBasler
from KalmanFilter import KalmanHomography
from Mosaicker import Mosaicker
from VideoRecorder import VideoRecorder
from RobotYumi import RobotYumi

def main():
    # Settings
    USE_YUMI = True # Toggle between real-time use or experiemnting with YuMi
    FILTER_MODE = 3  # 0=2D, 1=3D, 2=Depth, 3=Given
    SAVE_VIDEO = True
    SAVE_POSE = True # Note ground truth is only availible in YuMi mode
    SHOW_MOSAIC = True
    SHOW_APRILTAG_DETECTION = True

    # Initialise objects
    camera = CameraBasler(scale=0.5)
    apriltag_detector = DetectorApriltag()
    kalman = KalmanHomography()
    mosaicker = Mosaicker(shape=(2000, 2000))

    if USE_YUMI:
        yumi = RobotYumi()

    if SAVE_VIDEO:
        video_recorder = VideoRecorder("data/output_video", framerate=15., shape=(camera.h*2, camera.w))

    t_measured = []
    R_measured = []
    t_actual = []
    R_actual = []

    H_img_origin = None

    tic = time.time()
    i = 0

    # Begin process loop (ESC to exit)
    while camera.is_alive():
        # Step waypoint if in YuMi mode adn check when finished
        if USE_YUMI:
            if i >= len(yumi.waypoints) - 1:
                break
            i = yumi.step_waypoint()

        # Capture image and detect AprilTags and homography
        img = camera.capture_frame()
        results = apriltag_detector.detetct_tags(img)

        if results:
            H_img = apriltag_detector.calculate_homography(
                results, [0, 1, 2, 3, 4, 5, 6, 7])
        else:
            H_img = None

        kalman.predict_step(0.02)

        if H_img is not None:
            # Find Euclidian transform from homography
            R, t = homography_decomposition(H_img, camera.K)

            if SAVE_POSE:
                t_measured.append(t)
                R_measured.append(R)
                
                if USE_YUMI:
                    T_actual = yumi.robot.left.get_pose()
                    T_actual = T_actual.as_frames(
                        from_frame="world", to_frame="world")

                    t_actual.append((T_actual * yumi.R_origin.inverse() *
                                    yumi.gripper_offset.inverse() * yumi.R_origin).translation)
                    R_actual.append(T_actual.rotation)
                else:
                    t_actual.append(np.zeros(3))
                    R_actual.append(np.eye(3))

            # Calcualte warp for image to filter out swinging or other movement
            if FILTER_MODE == 0:
                H_euc = transfoms.crane_transform_2d(R, t, camera.K)
            elif FILTER_MODE == 1:
                H_euc = transfoms.crane_transform_3d(R, t, camera.K)
            elif FILTER_MODE == 2:
                H_euc = transfoms.crane_transform_depth(R, t, camera.K, 500.)
            elif FILTER_MODE == 3 and USE_YUMI:
                if H_img_origin is None:
                    H_img_origin = H_img
                    d_camera_origin = np.dot(np.matmul(R, t), np.array([0, 0, 1]))
                H_euc = transfoms.crane_transform_given(
                    yumi.t_boom[i], yumi.t_boom[0], d_camera_origin, H_img, H_img_origin, camera.K)
            else:
                H_euc = np.eye(3)

            # Pass homography through Kalman filter to remove high-frequency noise
            z = np.ndarray.flatten(np.matmul(H_euc, H_img))
            kalman.update_step(z)

            H_euc = np.matmul(np.reshape(
                kalman.x[:9], (3, 3)), np.linalg.inv(H_img))

            # Build mosaic for filtered image
            mosaicker.homography_mosaic(img, H_img, H_euc)

            cropped_mosaic = mosaicker.crop_mosaic(
                shape=(camera.w/1.25, camera.h/1.25))

            cropped_mosaic = cv2.resize(
                cropped_mosaic, (img.shape[1], img.shape[0]))

            if SAVE_VIDEO:
                video_recorder.write(cv2.hconcat([img, cropped_mosaic]))

            if SHOW_MOSAIC:
                cv2.imshow("mosaic", scale_img(mosaicker.mosaic, 0.5))

        # Show the output image after AprilTag detection
        if SHOW_APRILTAG_DETECTION:
            img = apriltag_detector.draw_detections(img, results)
            cv2.imshow("Image", scale_img(img, 0.5))

        # Exit on ESC
        key = cv2.waitKey(20)
        if key == 27:
            break

    if USE_YUMI:
        yumi.go_home()

    # Release resources
    camera.release()
    if SAVE_VIDEO:
        video_recorder.release()

    # Write pose to file
    if SAVE_POSE:
        with open('data/output_pose_raw.csv', mode='w') as f:
            writer = csv.writer(f, delimiter=',', quotechar='"',
                                quoting=csv.QUOTE_MINIMAL)

            writer.writerow(['x_m', 'y_m', 'z_m', 'R11_m', 'R12_m', 'R13_m', 'R21_m', 'R22_m', 'R23_m', 'R31_m', 'R32_m', 'R33_m',
                             'x_a', 'y_a', 'z_a', 'R11_a', 'R12_a', 'R13_a', 'R21_a', 'R22_a', 'R23_a', 'R31_a', 'R32_a', 'R33_a'])

            for i in range(len(t_measured)):
                writer.writerow(list(t_measured[i]) + list(R_measured[i].flatten()) + list(
                    t_actual[i]) + list(R_actual[i].flatten()))

        process_pose.main()


if __name__ == "__main__":
    main()
