import cv2
import numpy as np
import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from utils import *

# Create OpenCV window to continuously capture from webcam
vc = cv2.VideoCapture(0)

# Try to get the first frame
if vc.isOpened(): 
    rval, frame = vc.read()
else:
    rval = False

# Initialisation
prev_frame = None
H = np.eye(3)

# Plots for animated pose
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Marker and camera parameters
fx, fy, cx, cy = (739.2116337887949, 731.2693931923594, 472.1271812307942, 265.5094352085958)
K = extrinsic_matrix(fx, fy, cx, cy)

# Continuously capture frames
while rval:

    if prev_frame is not None:
        M, mask, kp1, kp2, matches = orb_homography(prev_frame, frame)
        H = M * H
        R, t = homography_decomposition(H, K)
        print(t)


        matchesMask = mask.ravel().tolist()

        h,w,_ = prev_frame.shape
        pts = np.float32([ [0, 0],[0, h-1],[w-1, h-1],[w-1, 0] ]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts,H)

        temp_frame = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        # Draw inliners
        draw_params = dict(matchColor = (0, 255, 0), # draw matches in green color
                        singlePointColor = None,
                        matchesMask = matchesMask, # draw only inliers
                        flags = 2)

        img3 = cv2.drawMatches(prev_frame, kp1, temp_frame, kp2, matches, None, **draw_params)

        cv2.imshow("compare", img3)

        x = t[0]
        y = t[1]
        z = t[2]

        plt.cla()
        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 500)
        ax.set_zlim(-500, 500)
        ax.quiver(x, y, z, R[0][0], R[1][0], R[2][0], length=50, color='r')
        ax.quiver(x, y, z, R[0][1], R[1][1], R[2][1], length=50, color='g')
        ax.quiver(x, y, z, R[0][2], R[1][2], R[2][2], length=50, color='b')

        plt.pause(0.01)


        warped_frame = cv2.warpPerspective(frame, (H), dsize=frame.shape[1::-1])
        cv2.imshow("Warped", warped_frame)

    prev_frame = frame

    # Display frame and capture new frame
    cv2.imshow("Image", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # Exit on ESC
        break

# Final cleanup
cv2.destroyWindow("preview")