import apriltag
import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation

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
    camera = CameraWebcam()
    apriltag_detector = DetectorApriltag()

    prev_img = None
    H = None
    H1 = np.eye(3)
    H2 = np.eye(3)

    mosaic_shape = (1000, 1000)
    mosaic = np.zeros((1000, 1000, 3), dtype='uint8')

    # Create OpenCV window to continuously capture from webcam
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    while camera.is_alive():
        img = camera.capture_frame()
        results = apriltag_detector.detetct_tags(img)

        # Plot the path travelled by the marker
        if results:
            H = apriltag_detector.calculate_homography(results, [0, 1, 2, 3])

        # elif prev_img is not None and H is not None:
        #     M = orb_homography(prev_img, img)

        #     if M is not None:
        #         H = M @ H

        if H is not None:
            R, t = homography_decomposition(H, camera.K)

            H4 = rigid_from_transform(R, t, camera.K)
            
            mosaic = homography_mosaic(mosaic, img, H1, H2, H, H4)

            H1 = H4
            H2 = H

            cv2.imshow("mosaic", mosaic)

            x = t[0]
            y = t[1]
            z = t[2]

            plt.cla()

            ax.set_xlim(-250, 250)
            ax.set_ylim(-250, 250)
            ax.set_zlim(0, 500)
            ax.quiver(x, y, z, R[0][0], R[1][0], R[2][0], length=50, color='r')
            ax.quiver(x, y, z, R[0][1], R[1][1], R[2][1], length=50, color='g')
            ax.quiver(x, y, z, R[0][2], R[1][2], R[2][2], length=50, color='b')
            
            plt.pause(0.01)

        prev_img = img
        # Show the output image after AprilTag detection
        img = apriltag_detector.draw_detections(img, results)
        cv2.imshow("Image", img)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    camera.release()

if __name__ == "__main__":
    main()