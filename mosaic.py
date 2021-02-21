import numpy as np
import cv2
import time

import apriltag
import time
from scipy.spatial.transform import Rotation

from utils import *
from transfoms import *
from DetectorApriltag import DetectorApriltag
from CameraWebcam import CameraWebcam
from CameraBasler import CameraBasler

def resize_img(img, scale):
    return cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

def homography_mosaic(img_src, img_dst, H1, H2, H3, H4):
    # img_src: Previous image with current mosaic
    # img_dst: New image to add to mosaic
    # H1: Homography used for previous frame to filter
    # H2: Homography for previous frame between AprilTag
    # H3: Homography for current frame between AprilTag
    # H4: Homography used for current frame to filter

    H_shift = np.array([
        [1, 0, img_src.shape[1] / 2 - img_dst.shape[1] / 2],
        [0, 1, img_src.shape[0] / 2 - img_dst.shape[0] / 2],
        [0, 0, 1]
    ])
    
    H_src_dst = np.matmul(H4, np.matmul(H3, np.matmul(np.linalg.inv(H2), np.linalg.inv(H1))))
    
    # # Alternative method
    # img_dst = cv2.warpPerspective(img_dst, H_shift @ H1 @ H2 @ np.linalg.inv(H3), dsize=img_src.shape[1::-1])
    # mosaic = make_mosaic(img_src, img_dst)
    # mosaic = cv2.warpPerspective(mosaic, H_shift @ H_src_dst @ np.linalg.inv(H_shift), dsize=img_src.shape[1::-1])
    
    img_src = cv2.warpPerspective(img_src, np.matmul(H_shift, np.matmul(H_src_dst, np.linalg.inv(H_shift))), dsize=img_src.shape[1::-1], flags=cv2.INTER_NEAREST)
    img_dst = cv2.warpPerspective(img_dst, np.matmul(H_shift, H4), dsize=img_src.shape[1::-1], flags=cv2.INTER_NEAREST)
    mosaic = make_mosaic(img_src, img_dst)

    return mosaic



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

    # Sharpen image edges to remove blurring caused through image warping
    kernal = np.ones((5, 5), np.uint8)
    # mask0 = cv2.erode(mask0, kernal, iterations=1)
    mask1 = cv2.erode(mask1, kernal, iterations=1)

    # img0 = cv2.bitwise_and(img0, img0, mask=mask0)
    img1 = cv2.bitwise_and(img1, img1, mask=mask1)
    
    # Find overlapping regions of both images to blend together
    mask = cv2.bitwise_and(mask0, mask1)

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

    img1 = cv2.imread('images/0.png')
    img2 = cv2.imread('images/2.png')
    img3 = cv2.imread('images/1.png')

    K = np.array([[1.65589856e+03, 0.00000000e+00, 7.45973980e+02],
                  [0.00000000e+00, 1.65939697e+03, 1.05925386e+03],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    cx = int(K[0, 2])
    cy = int(K[0, 1])

    M = cv2.getRotationMatrix2D((cx, cy), -10.0, 1.0)
    img1 = cv2.warpAffine(img1, M, img1.shape[1::-1])
    mosaic_shape = (1000, 1000)

    apriltag_detector = DetectorApriltag()

    results1 = apriltag_detector.detetct_tags(img1)
    results2 = apriltag_detector.detetct_tags(img2)
    results3 = apriltag_detector.detetct_tags(img3)

    H2 = apriltag_detector.calculate_homography(results1, [0, 1, 2, 3])
    H3 = apriltag_detector.calculate_homography(results2, [0, 1, 2, 3])
    H3_p = apriltag_detector.calculate_homography(results3, [0, 1, 2, 3])

    R1, t1 = homography_decomposition(H2, K)
    R2, t2 = homography_decomposition(H3, K)
    R3, t3 = homography_decomposition(H3_p, K)

    img1, H1 = filter_swinging(img1, R1, t1, K)
    temp, H4 = filter_swinging(img2, R2, t2, K)
    temp, H4_p = filter_swinging(img3, R3, t3, K)

    # cv2.imshow("1", resize_img(img1, 0.25))
    # cv2.waitKey(0)
    # img1 = cv2.warpPerspective(img1, np.linalg.inv(H1), dsize=img1.shape[1::-1])
    # cv2.imshow("2", resize_img(img1, 0.25))
    # cv2.waitKey(0)
    # img1 = cv2.warpPerspective(img1, H3 @ np.linalg.inv(H2), dsize=img1.shape[1::-1])
    # cv2.imshow("3", resize_img(img1, 0.25))
    # cv2.waitKey(0)

    # H_shift = np.array([
    #     [1, 0, mosaic_shape[0]/2.-cx],
    #     [0, 1, mosaic_shape[1]/2.-cy],
    #     [0, 0, 1]
    # ])

    # H_ultra = H_shift @ H3 @ np.linalg.inv(H2) @ np.linalg.inv(H1)
    # img1 = cv2.warpPerspective(img1, H_ultra, dsize=mosaic_shape)
    # img2 = cv2.warpPerspective(img2, H_shift, dsize=mosaic_shape)

    # mosaic = make_mosaic(img1, img2)
    # # cv2.imshow("4", resize_img(mosaic, 0.25))
    # mosaic = homography_mosaic(img1, img2, H1, H2, H3, H4)
    # cv2.imshow("5", resize_img(mosaic, 0.25))
    # cv2.imshow("5", resize_img(homography_mosaic(mosaic, img3, H4, H3, H3_p, H4_p), 0.25))


    mosaic = np.zeros((4000, 4000, 3), dtype='uint8')

    H1 = np.eye(3)
    H2 = np.eye(3)

    mosaic = homography_mosaic(mosaic, img1, np.eye(3), np.eye(3), H2, H1)
    cv2.imshow("4", resize_img(mosaic, 0.25))
    mosaic = homography_mosaic(mosaic, img2, H1, H2, H3, H4)
    cv2.imshow("5", resize_img(mosaic, 0.25))
    mosaic = homography_mosaic(mosaic, img3, H4, H3, H3_p, H4_p)
    cv2.imshow("6", resize_img(mosaic, 0.25))
       



    cv2.waitKey(0)