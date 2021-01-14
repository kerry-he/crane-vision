import apriltag
import cv2
import numpy as np

from utils import *

# Setup AprilTag detector
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)


tag_size = 38.0
tag_corners = np.array([
    [-tag_size / 2, -tag_size / 2], 
    [tag_size / 2, -tag_size / 2], 
    [tag_size / 2, tag_size / 2], 
    [-tag_size / 2, tag_size / 2]
])

src_corners = [
    tag_corners + [-57.0, -97.0],
    tag_corners + [57.0, -97.0],
    tag_corners + [-57.0, 97.0],
    tag_corners + [57.0, 97.0]
]

mosaic_shape = (1000, 1000)
src_corners = src_corners + np.asarray(mosaic_shape) / 2

def apriltag_homography(img):

    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    results = detector.detect(img)

    src = np.empty((0, 2))
    dst = np.empty((0, 2))

    # Loop over the AprilTag detection results
    for r in results:
        if r.tag_id >= 0 and r.tag_id <=3:
            src = np.append(src, src_corners[r.tag_id], axis=0)
            dst = np.append(dst, r.corners, axis=0)
        
    # Plot the path travelled by the marker
    if results:
        H, _ = cv2.findHomography(np.asarray(src), np.asarray(dst), cv2.RANSAC, 10.0)
        return H

    return None

def apriltag_points(img, id0, idN):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    results = detector.detect(img)

    ptns = np.empty((0, 2))

    # Loop over the AprilTag detection results
    for r in results:
        if r.tag_id >= id0 and r.tag_id <= idN:
            ptns = np.append(ptns, r.corners, axis=0)

    return ptns