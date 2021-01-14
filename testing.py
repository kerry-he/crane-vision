import apriltag
import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation

from utils import *
from kalman_filter import *
from apriltag_util import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Setup AprilTag detector
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)
path = []

# Marker and camera parameters
fx, fy, cx, cy = (739.2116337887949, 731.2693931923594, 472.1271812307942, 265.5094352085958)
K = extrinsic_matrix(fx, fy, cx, cy)

img0 = cv2.imread('images/test2.jpg',0)
img1 = cv2.imread('images/test3.jpg',0)

# Method 1
H0 = apriltag_homography(img0)
H1 = apriltag_homography(img1)

H = H1 @ np.linalg.inv(H0)

print(H)
print(homography_decomposition(H, K))

ptns0 = apriltag_points(img0, 0, 3)
ptns1 = apriltag_points(img1, 0, 3)

H, _ = cv2.findHomography(ptns0, ptns1, cv2.RANSAC, 10.0)

warped0 = cv2.warpPerspective(img0, np.linalg.inv(H0), dsize=mosaic_shape)
warped1 = cv2.warpPerspective(img1, np.linalg.inv(H1), dsize=mosaic_shape)

cv2.imshow("0", warped0)
cv2.imshow("1", warped1)

tic = time.time()

ret, mask0 = cv2.threshold(warped0, 1, 255, cv2.THRESH_BINARY)
ret, mask1 = cv2.threshold(warped1, 1, 255, cv2.THRESH_BINARY)

mask = cv2.bitwise_and(mask0, mask1)

cropped0 = cv2.bitwise_and(warped0, warped0, mask=mask)
cropped1 = cv2.bitwise_and(warped1, warped1, mask=mask)

remainder0 = cv2.bitwise_and(warped0, warped0, mask=cv2.bitwise_not(mask))
remainder1 = cv2.bitwise_and(warped1, warped1, mask=cv2.bitwise_not(mask))

mosaic = cv2.addWeighted(cropped0, 0.5, cropped1, 0.5, 0)
mosaic = cv2.add(mosaic, remainder0)
mosaic = cv2.add(mosaic, remainder1)

print(time.time() - tic)

cv2.imshow("both", mosaic)


cv2.waitKey(0)