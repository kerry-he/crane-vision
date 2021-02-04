from pypylon import pylon
from pypylon import genicam
import cv2

from utils import *


class CameraBasler():
    def __init__(self):
        # Conecting to the first available camera
        self.camera = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()

        # Load in camera parameters
        nodeFile = "NodeMap.pfs"
        pylon.FeaturePersistence.Load(nodeFile, self.camera.GetNodeMap(), True)

        # Grabing Continusely (video) with minimal delay
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter = pylon.ImageFormatConverter()

        # Converting to opencv bgr format
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        self.h = self.camera.Height.GetValue()
        self.w = self.camera.Width.GetValue()

        # Marker and camera parameters
        fx, fy, cx, cy = (1.79783924e+03, 1.79734316e+03,
                          1.00904154e+03, 7.47131925e+02) #2064, 1544
        K = extrinsic_matrix(fx, fy, cx, cy)

        self.distortion = np.asarray([[-1.73606123e-01, 8.51898824e-02,
                                       1.64237962e-04, -7.81281736e-04, -8.26497994e-03]])

        # Perform calculations to obtain intrisic (K) and distortion (dist) parameters
        self.K, _ = cv2.getOptimalNewCameraMatrix(
            K, self.distortion, (self.w, self.h), 1.0, (self.w, self.h))
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(
            K, self.distortion, None, self.K, (self.w, self.h), 5)

        # Rotate intrinsic parameter K to account for 90CCW rotation
        self.K = extrinsic_matrix(self.K[1, 1], self.K[0, 0], self.K[1, 2], 2064-self.K[0, 2])
        print(self.K)

    def capture_frame(self):
        grabResult = self.camera.RetrieveResult(
            5000, pylon.TimeoutHandling_ThrowException)

        img = None

        if grabResult.GrabSucceeded():
            # Access the image data
            image = self.converter.Convert(grabResult)
            img = image.GetArray()

            # Undistort camera frame
            img = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        grabResult.Release()
        return img

    def is_alive(self):
        return self.camera.IsGrabbing()

    def release(self):
        # Releasing the resource
        self.camera.StopGrabbing()