##################################################################
# CALIBRATE CAMERA
# Script which reads images from a specified folder and calculates
# the extrinsic camera parameters and distortion parameters.
##################################################################

# Libraries
import numpy as np
import cv2
import os

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

patternsize = (7, 10)
objp = np.zeros((patternsize[0] * patternsize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:patternsize[0], 0:patternsize[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

folder = "data/calibration"
images = os.listdir(folder)

for i, fname in enumerate(images):
    print("Processing image %d / %d" % (i + 1, len(images)))
    img = cv2.imread(folder + "/" + fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, patternsize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print("Found chessboard pattern in image %d / %d" % (i + 1, len(images)))
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, patternsize, corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

h, w = img.shape[:2]

# Perform camera calibration to obtain intrisic (mtx) and distortion (dist) parameters
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

fx = mtx[0,0]
fy = mtx[1,1]
cx = mtx[0,2]
cy = mtx[1,2]

params = (fx, fy, cx, cy)

print()
print('Intrinsic parameters')
print('  fx, fy, cx, cy = {}'.format(repr(params)))
print()
print("Distortion parameters")
print('  k1 k2 p1 p2 k3 = {}'.format(repr(dist)))

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1.0, (w,h))


# Generate undistortion map - remapping using this map will be faster than undistorting directly
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

cv2.imshow('img',img)
cv2.imshow('dst',dst)

cv2.waitKey(0)

cv2.destroyAllWindows()