import cv2

from utils import *


class CameraWebcam():
    def __init__(self):
        # Conecting to the first available camera
        self.vc = cv2.VideoCapture(0)

        # Define camera parameters
        self.rval, img = self.vc.read()
        self.h, self.w = img.shape[:2]

        fx, fy, cx, cy = (739.2116337887949, 731.2693931923594,
                          472.1271812307942, 265.5094352085958)
        K = extrinsic_matrix(fx, fy, cx, cy)

        self.distortion = np.asarray(
            [[0.05610004, -0.41182393, -0.00226416,  0.00479014,  0.67998375]])

        # Perform calculations to obtain intrisic (K) and distortion (dist) parameters
        self.K, _ = cv2.getOptimalNewCameraMatrix(
            K, self.distortion, (self.w, self.h), 1.0, (self.w, self.h))
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            K, self.distortion, None, self.K, (self.w, self.h), 5)

    def capture_frame(self):
        self.rval, img = self.vc.read()

        if self.rval:
            # Undistort camera frame
            img = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)
            return img

        return None

    def is_alive(self):
        return self.rval

    def release(self):
        pass