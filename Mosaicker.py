##################################################################
# MOSAICKER
# Class to perform mosaicking and iterative image stitching
# functinalities.
##################################################################

# Libraries
import numpy as np
import cv2

class Mosaicker():

    def __init__(self, shape):
        self.shape = shape
        self.mosaic = np.zeros((shape[0], shape[1], 3), dtype='uint8')

        # H_euc_prev: Homography used for previous frame to filter
        # H_img_prev: Homography for previous frame between AprilTag
        self.H_img_prev = np.eye(3)
        self.H_euc_prev = np.eye(3)

    def homography_mosaic(self, img, H_img, H_euc):
        # Iterative mosaicking function which builds on previous image to build mosaic
        # in new camera perspective

        # img: New image to add to mosaic
        # H_img: Homography for current frame between AprilTag
        # H_euc: Homography used for current frame to filter

        # Homography required to shift new image to center of mosaic
        H_shift = np.array([
            [1, 0, self.shape[1] / 2 - img.shape[1] / 2],
            [0, 1, self.shape[0] / 2 - img.shape[0] / 2],
            [0, 0, 1]
        ])

        # Calculate net homography required to transform previous mosaic
        H_src_dst = np.matmul(H_euc, np.matmul(
            H_img, np.matmul(np.linalg.inv(self.H_img_prev), np.linalg.inv(self.H_euc_prev))))

        H_src_dst = np.matmul(H_shift, np.matmul(
            H_src_dst, np.linalg.inv(H_shift)))

        # Warp images before stitching together
        self.mosaic = cv2.warpPerspective(
            self.mosaic, H_src_dst, dsize=self.shape, flags=cv2.INTER_NEAREST)

        img = cv2.warpPerspective(img, np.matmul(
            H_shift, H_euc), dsize=self.shape, flags=cv2.INTER_NEAREST)

        self.mosaic = make_mosaic(self.mosaic, img)

        # Update previous homographies
        self.H_img_prev = H_img
        self.H_euc_prev = H_euc

        return self.mosaic

    def crop_mosaic(self, shape):
        return self.mosaic[int(self.shape[0]//2 - shape[0]//2): int(self.shape[0]//2 + shape[0]//2),
                           int(self.shape[1]//2 - shape[1]//2): int(self.shape[1]//2 + shape[1]//2)]


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
    _, mask0 = cv2.threshold(gray0, 1, 255, cv2.THRESH_BINARY)
    _, mask1 = cv2.threshold(gray1, 1, 255, cv2.THRESH_BINARY)

    # Sharpen image edges to remove blurring caused through image warping
    kernal = np.ones((5, 5), np.uint8)
    mask1 = cv2.erode(mask1, kernal, iterations=1)
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
