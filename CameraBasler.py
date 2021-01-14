from pypylon import pylon
from pypylon import genicam
import cv2
import time


class CameraBasler():
    def __init__(self):
        # Conecting to the first available camera
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
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

        # Marker and camera parameters
        fx, fy, cx, cy = (1.79783924e+03, 1.79734316e+03, 1.00904154e+03, 7.47131925e+02)
        mtx = extrinsic_matrix(fx, fy, cx, cy)
        dist = np.asarray([[-1.73606123e-01, 8.51898824e-02, 1.64237962e-04, -7.81281736e-04, -8.26497994e-03]])


    def capture_frame(self):
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        

while camera.IsGrabbing():

    

    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()

        cv2.namedWindow('title', cv2.WINDOW_NORMAL)
        cv2.imshow('title', img)
        k = cv2.waitKey(1)
        if k == 27:
            break
    grabResult.Release()
    
# Releasing the resource    
camera.StopGrabbing()

cv2.destroyAllWindows()