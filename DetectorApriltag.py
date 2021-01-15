import apriltag
import cv2
import numpy as np

from utils import *

class DetectorApriltag():
    def __init__(self):

        # Setup AprilTag detector
        options = apriltag.DetectorOptions(families="tag36h11")
        self.detector = apriltag.Detector(options)

        tag_size = 38.0
        tag_corners = np.array([
            [-tag_size / 2, -tag_size / 2], 
            [tag_size / 2, -tag_size / 2], 
            [tag_size / 2, tag_size / 2], 
            [-tag_size / 2, tag_size / 2]
        ])

        self.src_corners = [
            tag_corners + [-57.0, -97.0],
            tag_corners + [57.0, -97.0],
            tag_corners + [-57.0, 97.0],
            tag_corners + [57.0, 97.0]
        ]

        mosaic_shape = (1000, 1000)
        self.src_corners += np.asarray(mosaic_shape) / 2

    def detetct_tags(self, img):
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return self.detector.detect(img)

    def calculate_homography(self, results, ids):
        if results is None:
            return None

        src = np.empty((0, 2))
        dst = np.empty((0, 2))

        # Loop over the AprilTag detection results
        for r in results:
            if r.tag_id in ids:
                src = np.append(src, self.src_corners[r.tag_id], axis=0)
                dst = np.append(dst, r.corners, axis=0)
            
        # Plot the path travelled by the marker
        H, _ = cv2.findHomography(np.asarray(src), np.asarray(dst), cv2.RANSAC, 10.0)
        return H

    def draw_detections(self, img, results):
        for r in results:
            # Extract the bounding box (x, y)-coordinates for the AprilTag
            # and convert each of the (x, y)-coordinate pairs to integers
            (ptA, ptB, ptC, ptD) = r.corners
            ptA = (int(ptA[0]), int(ptA[1]))
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))
            
            # Draw the bounding box of the AprilTag detection
            cv2.line(img, ptA, ptB, (0, 128, 0), 2)
            cv2.line(img, ptB, ptC, (255, 0, 0), 2)
            cv2.line(img, ptC, ptD, (0, 0, 255), 2)
            cv2.line(img, ptD, ptA, (255, 255, 0), 2)

            # Draw the center (x, y)-coordinates of the AprilTag
            (cX, cY) = (int(r.center[0]), int(r.center[1]))
            cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)

            # Draw the tag family on the image
            cv2.putText(img, str(r.tag_id), (ptA[0], ptA[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img