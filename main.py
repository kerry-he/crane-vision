import apriltag
import cv2

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Setup AprilTag detector
options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)
path = []

# Marker and camera parameters
fx, fy, cx, cy = (739.2116337887949, 731.2693931923594, 472.1271812307942, 265.5094352085958)
tag_size = 45

# Create OpenCV window to continuously capture from webcam
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = detector.detect(gray)

    # loop over the AprilTag detection results
    for r in results:
        M, _, _ = detector.detection_pose(r, (fx, fy, cx, cy), tag_size)
        print(M[:3, 3])
        path.append(M[:3, 3])

        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptA = (int(ptA[0]), int(ptA[1]))
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        
        # draw the bounding box of the AprilTag detection
        cv2.line(frame, ptA, ptB, (0, 255, 0), 2)
        cv2.line(frame, ptB, ptC, (0, 255, 0), 2)
        cv2.line(frame, ptC, ptD, (0, 255, 0), 2)
        cv2.line(frame, ptD, ptA, (0, 255, 0), 2)

        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)

        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(frame, tagFamily, (ptA[0], ptA[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("[INFO] tag family: {}".format(tagFamily))
        
    # show the output image after AprilTag detection
    cv2.imshow("Image", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
cv2.destroyWindow("preview")

# Plot the path travelled by the marker
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = [i[0] for i in path]
y = [i[1] for i in path]
z = [i[2] for i in path]

ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()