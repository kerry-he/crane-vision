import apriltag
import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation

from utils import *
from kalman_filter import *

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Setup Basler camera
from pypylon import pylon
from pypylon import genicam
import cv2
import time

# Conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Load in camera parameters
nodeFile = "NodeMap.pfs"
pylon.FeaturePersistence.Load(nodeFile, camera.GetNodeMap(), True)

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()

# Converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

h = camera.Height.GetValue()
w = camera.Width.GetValue()

# Setup AprilTag detector
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)
path = []


# Marker and camera parameters
fx, fy, cx, cy = (1.79783924e+03, 1.79734316e+03, 1.00904154e+03, 7.47131925e+02)
mtx = extrinsic_matrix(fx, fy, cx, cy)
dist = np.asarray([[-1.73606123e-01, 8.51898824e-02, 1.64237962e-04, -7.81281736e-04, -8.26497994e-03]])

# Perform camera calibration to obtain intrisic (mtx) and distortion (dist) parameters
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1.0, (w,h))
K = newcameramtx
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)


tag_size = 38.0
tag_corners = np.array([[-tag_size / 2, -tag_size / 2], [tag_size / 2, -tag_size / 2], [tag_size / 2, tag_size / 2], [-tag_size / 2, tag_size / 2]])

src_corners = [
    tag_corners + [-57.0, -97.0],
    tag_corners + [57.0, -97.0],
    tag_corners + [-57.0, 97.0],
    tag_corners + [57.0, 97.0]
]

kalman = Kalman()

prev_frame = None
H = None

mosaic_shape = (1000, 1000)
mosaic_frame = np.zeros((1000, 1000, 3), dtype='uint8')

# Profiling variables
profile_x = [[], [], [], []]


# Create OpenCV window to continuously capture from webcam
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

src_corners = src_corners + np.asarray(mosaic_shape) / 2

while camera.IsGrabbing():

    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        tic = time.time()

        # Access the image data
        image = converter.Convert(grabResult)
        frame = image.GetArray()
        frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = detector.detect(gray)

        src = np.empty((0, 2))
        dst = np.empty((0, 2))

        # loop over the AprilTag detection results
        for r in results:

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
            # print("[INFO] tag family: {}".format(tagFamily))

            if r.tag_id >= 0 and r.tag_id <=3:
                src = np.append(src, src_corners[r.tag_id], axis=0)
                dst = np.append(dst, r.corners, axis=0)

        toc = time.time()
        kalman.predict_step(toc - tic)
            
        # Plot the path travelled by the marker
        if results:
            H, _ = cv2.findHomography(np.asarray(src), np.asarray(dst), cv2.RANSAC, 10.0)

            warped_frame = cv2.warpPerspective(frame, np.linalg.inv(H), dsize=mosaic_shape)
            mosaic_frame = make_mosaic(mosaic_frame, warped_frame)
            cv2.imshow("mosaic", mosaic_frame)

        # elif prev_frame is not None and H is not None:
        #     M, mask, kp1, kp2, matches = orb_homography(prev_frame, frame)

        #     if M is not None:
        #         H = M @ H

        
        if H is not None:
            R, t = homography_decomposition(H, K)
            
            # warped_frame = cv2.warpPerspective(frame, np.linalg.inv(H), dsize=mosaic_shape)
            # mosaic_frame = make_mosaic(mosaic_frame, warped_frame)
            # cv2.imshow("mosaic", mosaic_frame)
            
                
            # r = Rotation.from_matrix(R)
            # z = np.concatenate((np.squeeze(t), r.as_euler('xyz', degrees=True)))
            # kalman.update_step(z)
            # profile_x[len(results) - 1].append(z)

            # x = kalman.x[0]
            # y = kalman.x[1]
            # z = kalman.x[2]

            # r = Rotation.from_euler('xyz', kalman.x[3:6], degrees=True)
            # R = r.as_matrix()

            x = t[0]
            y = t[1]
            z = t[2]

            print(t)

            plt.cla()
            ax.set_xlim(-250, 250)
            ax.set_ylim(-250, 250)
            ax.set_zlim(0, 500)
            ax.quiver(x, y, z, R[0][0], R[1][0], R[2][0], length=50, color='r')
            ax.quiver(x, y, z, R[0][1], R[1][1], R[2][1], length=50, color='g')
            ax.quiver(x, y, z, R[0][2], R[1][2], R[2][2], length=50, color='b')

            plt.pause(0.01)

        prev_frame = frame

        # show the output image after AprilTag detection
        cv2.imshow("Image", frame)
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

# Releasing the resource    
camera.StopGrabbing()

cv2.destroyAllWindows()

for x in profile_x:
    x = np.asarray(x)
    print(x.shape)
    print(np.std(x, axis=0))