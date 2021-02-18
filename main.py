import apriltag
import cv2
import numpy as np
import time
import csv
from scipy.spatial.transform import Rotation
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

def main():
    camera = CameraBasler(scale=0.5)
    apriltag_detector = DetectorApriltag()
    kalman = KalmanHomography()

    prev_img = None
    H = None
    H1 = np.eye(3)
    H2 = np.eye(3)

    H_origin = None
    d_camera_origin = None

    mosaic_shape = (2000, 2000)
    mosaic = np.zeros((2000, 2000, 3), dtype='uint8')

    _name = "output" + '.mp4'
    _fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    _out = cv2.VideoWriter(_name, _fourcc, 15.0, (camera.h*2,camera.w))

    t_history = []
    R_history = []
    H_history = []

    # Create OpenCV window to continuously capture from webcam
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    tic = time.time()

    while camera.is_alive():
        img = camera.capture_frame()
        results = apriltag_detector.detetct_tags(img)

        # cv2.drawMarker(img, (746, 1059),  (255, 0, 0), cv2.MARKER_CROSS, 50, 3)

        # Plot the path travelled by the marker
        tic_test = time.time()
        if results:
            H3 = apriltag_detector.calculate_homography(results, [0, 1, 2, 3, 4, 5, 6, 7])
        else:
            H3 = None

        # elif prev_img is not None and H is not None:
        #     M = orb_homography(prev_img, img)

        #     if M is not None:
        #         H = M @ H

        toc = time.time()
        kalman.predict_step(toc - tic)
        tic = time.time()

        if H3 is not None:
            R, t = homography_decomposition(H3, camera.K)

            if H_origin is None:
                H_origin = H3
                d_camera_origin = np.dot(R @ t, np.array([0, 0, 1]))

            # c = np.matmul(H3, np.array([0., 0., 1.]))
            # cv2.circle(img, (int(c[0]), int(c[1])), 5, (0, 0, 255), -1)
            # print(t)

            # z = np.concatenate((np.squeeze(t), transforms3d.euler.mat2euler(R)))
            # kalman.update_step(z)

            # t = kalman.x[:3]
            # R = transforms3d.euler.euler2mat(kalman.x[3], kalman.x[4], kalman.x[5])

            # H4 = crane_transform_2d(R, t, camera.K)
            H4 = crane_transform_3d(R, t, camera.K)
            # H4 = crane_transform_given(None, None, camera.K, d_camera_origin)
            # H4 = crane_transform_depth(R, t, camera.K, 500.)
            # H4 = H4 @ H_origin @ np.linalg.inv(H3)

            H_src_dst = H4 @ H3
            H_history.append(np.ndarray.flatten(H_src_dst))

            z = np.ndarray.flatten(H_src_dst)
            kalman.update_step(z)   

            H4 = np.reshape(kalman.x[:9], (3, 3)) @ np.linalg.inv(H3)

            
            mosaic = homography_mosaic(mosaic, img, H1, H2, H3, H4)

            H1 = H4
            H2 = H3
            cropped_mosaic = mosaic[1000-int(camera.w/2.5):1000+int(camera.w/2.5), 1000-int(camera.h/2.5):1000+int(camera.h/2.5)]
            cropped_mosaic = cv2.resize(cropped_mosaic, (img.shape[1], img.shape[0]))
            warped_img = cv2.warpPerspective(img, H4, dsize=img.shape[1::-1])

            _out.write(cv2.hconcat([img, cropped_mosaic]))

            # cv2.imshow("mosaic", cv2.resize(cropped_mosaic, (cropped_mosaic.shape[1]//2, cropped_mosaic.shape[0]//2))) #cv2.resize(mosaic, (5000 // 4, 5000 // 4))
            cv2.imshow("mosaic", cv2.resize(mosaic, (2000 // 2, 2000 // 2)))
            # cv2.imshow("warped", cv2.resize(warped_img, (img.shape[1]//2, img.shape[0]//2)))

            t_history.append(t)
            R_history.append(Rotation.from_matrix(R).as_euler('xyz'))

            # x = t[0]
            # y = t[1]
            # z = t[2]

            # plt.cla()

            # ax.set_xlim(-250, 250)
            # ax.set_ylim(-250, 250)
            # ax.set_zlim(0, 500)
            # ax.quiver(x, y, z, R[0][0], R[1][0], R[2][0], length=50, color='r')
            # ax.quiver(x, y, z, R[0][1], R[1][1], R[2][1], length=50, color='g')
            # ax.quiver(x, y, z, R[0][2], R[1][2], R[2][2], length=50, color='b')
            
            # plt.pause(0.01)

        prev_img = img
        # Show the output image after AprilTag detection
        # img = apriltag_detector.draw_detections(img, results)
        # cv2.imshow("Image", cv2.resize(img, (int(1544/2), int(2064/2))))
        
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    camera.release()
    _out.release()
    
    # Write results to file
    with open('results.csv', mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['x', 'y', 'z', 'Rx', 'Ry', 'Rz'])
        for i in range(len(t_history)):
            writer.writerow(list(t_history[i]) + list(R_history[i]) + list(H_history[i]))

if __name__ == "__main__":
    main()