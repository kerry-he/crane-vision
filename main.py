import apriltag
import cv2
import numpy as np
from utils import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Setup AprilTag detector
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)
path = []

# Marker and camera parameters
fx, fy, cx, cy = (739.2116337887949, 731.2693931923594, 472.1271812307942, 265.5094352085958)
tag_size = 45.0
srcPoints = np.array([[-tag_size / 2, -tag_size / 2], [tag_size / 2, -tag_size / 2], [tag_size / 2, tag_size / 2], [-tag_size / 2, tag_size / 2]])
K = extrinsic_matrix(fx, fy, cx, cy)

# Create OpenCV window to continuously capture from webcam
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = detector.detect(gray)

    # loop over the AprilTag detection results
    for r in results:
        M, _, _ = detector.detection_pose(r, (fx, fy, cx, cy), tag_size)
        # print(M[:3, :3])
        # print(M[:3, 3])
        path.append(M[:3, 3])

        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        
        # draw the bounding box of the AprilTag detection
        cv2.line(frame, ptA, ptB, (0, 128, 0), 2)
        cv2.line(frame, ptB, ptC, (255, 0, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 0, 255), 2)
        cv2.line(frame, ptD, ptA, (255, 255, 0), 2)

        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(frame, tagFamily, (ptA[0], ptA[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("[INFO] tag family: {}".format(tagFamily))

        H, _ = cv2.findHomography(srcPoints, np.asarray(r.corners))
        H_ = calculate_homography(srcPoints, np.asarray(r.corners))
        R, t = homography_decomposition(H, K)
        R_, t_ = homography_decomposition(H_, K)

        print(R, t)
        print(R_, t_)
        # print(R, t)
        
    # Plot the path travelled by the marker
    if results:
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

    # show the output image after AprilTag detection
    cv2.imshow("Image", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")