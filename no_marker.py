import cv2
import numpy as np
import time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Create OpenCV window to continuously capture from webcam
vc = cv2.VideoCapture(0)

# Try to get the first frame
if vc.isOpened(): 
    rval, frame = vc.read()
else:
    rval = False

# Continuously capture frames
while rval:



    cv2.imshow("Image", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # Exit on ESC
        break

# Final cleanup
cv2.destroyWindow("preview")