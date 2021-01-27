import numpy as np
import cv2

import apriltag
import time
from scipy.spatial.transform import Rotation

from utils import *
from kalman_filter import *
from transfoms import *
from DetectorApriltag import DetectorApriltag
from CameraWebcam import CameraWebcam
from CameraBasler import CameraBasler

def homography_mosaic(img_src, img_dst, H1, H2, H3, H4):
    # img_src: Previous image with current mosaic
    # img_dst: New image to add to mosaic
    # H1: Homography used for previous frame to filter
    # H2: Homography for previous frame between AprilTag
    # H3: Homography for current frame between AprilTag
    # H4: Homography used for current frame to filter

    H_shift = np.array([
        [1, 0, mosaic_shape[0]/2.-cx],
        [0, 1, mosaic_shape[1]/2.-cy],
        [0, 0, 1]
    ])

    H_src_dst = H3 @ np.linalg.inv(H2) @ np.linalg.inv(H1)

    img_src = cv2.warpPerspective(img_src, H_src_dst, dsize=mosaic_shape)
    img2 = cv2.warpPerspective(img2, H_shift, dsize=mosaic_shape)

    mosaic = make_mosaic(img1, img2)

    return cv2.warpPerspective(mosaic, H4, dsize=mosaic_shape)



def make_mosaic(img0, img1):

    # Convert images to B&W
    if len(img0.shape) > 2:
        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    else:
        gray0 = img0
    if len(img1.shape) > 2:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1

    # Obtain mask of warped images to find overlapping region
    ret, mask0 = cv2.threshold(gray0, 1, 255, cv2.THRESH_BINARY)
    ret, mask1 = cv2.threshold(gray1, 1, 255, cv2.THRESH_BINARY)
    
    mask = cv2.bitwise_and(mask0, mask1)

    # Find overlapping regions of both images to blend together
    cropped0 = cv2.bitwise_and(img0, img0, mask=mask)
    cropped1 = cv2.bitwise_and(img1, img1, mask=mask)

    # Find remaining area to add on later
    remainder0 = cv2.bitwise_and(img0, img0, mask=cv2.bitwise_not(mask))
    remainder1 = cv2.bitwise_and(img1, img1, mask=cv2.bitwise_not(mask))

    # Combine images
    mosaic = cv2.addWeighted(cropped0, 0.5, cropped1, 0.5, 0)
    mosaic = cv2.add(mosaic, remainder0)
    mosaic = cv2.add(mosaic, remainder1)

    return mosaic



if __name__ == "__main__":

    img1 = cv2.imread('images/test0.jpg',0)
    img2 = cv2.imread('images/test1.jpg',0)

    fx, fy, cx, cy = (739.2116337887949, 731.2693931923594,
                        320.0, 240.0)
    K = extrinsic_matrix(fx, fy, cx, cy)

    M = cv2.getRotationMatrix2D((cx, cy), -10.0, 1.0)
    img1 = cv2.warpAffine(img1, M, img1.shape[1::-1])
    mosaic_shape = (1000, 1000)

    apriltag_detector = DetectorApriltag()

    results1 = apriltag_detector.detetct_tags(img1)
    results2 = apriltag_detector.detetct_tags(img2)

    H2 = apriltag_detector.calculate_homography(results1, [0, 1, 2, 3])
    H3 = apriltag_detector.calculate_homography(results2, [0, 1, 2, 3])

    R1, t1 = homography_decomposition(H2, K)
    R2, t2 = homography_decomposition(H3, K)

    print(R1, t1)
    print(R2, t2)

    img1, H1 = filter_swinging(img1, R1, t1, K)
    test_frame, H4 = filter_swinging(img2, R2, t2, K)

    # cv2.imshow("1", img1)
    # cv2.waitKey(0)
    # img1 = cv2.warpPerspective(img1, np.linalg.inv(H1), dsize=img1.shape[1::-1])
    # cv2.imshow("2", img1)
    # cv2.waitKey(0)
    # img1 = cv2.warpPerspective(img1, H3 @ np.linalg.inv(H2), dsize=img1.shape[1::-1])
    # cv2.imshow("3", img1)
    # cv2.waitKey(0)

    H_shift = np.array([
        [1, 0, mosaic_shape[0]/2.-cx],
        [0, 1, mosaic_shape[1]/2.-cy],
        [0, 0, 1]
    ])

    H_ultra = H_shift @ H3 @ np.linalg.inv(H2) @ np.linalg.inv(H1)
    img1 = cv2.warpPerspective(img1, H_ultra, dsize=mosaic_shape)
    img2 = cv2.warpPerspective(img2, H_shift, dsize=mosaic_shape)

    mosaic = make_mosaic(img1, img2)
    cv2.imshow("3", mosaic)
    cv2.imshow("4", img1)
    cv2.waitKey(0)