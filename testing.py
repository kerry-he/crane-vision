import apriltag
import cv2

img = cv2.imread('images/sample1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detector = apriltag.Detector()
results = detector.detect(gray)
print(results)

fx, fy, cx, cy = (739.2116337887949, 731.2693931923594, 472.1271812307942, 265.5094352085958)
print(detector.detection_pose(results[0], (fx, fy, cx, cy)))

# loop over the AprilTag detection results
for r in results:
	# extract the bounding box (x, y)-coordinates for the AprilTag
	# and convert each of the (x, y)-coordinate pairs to integers
	(ptA, ptB, ptC, ptD) = r.corners
	ptB = (int(ptB[0]), int(ptB[1]))
	ptC = (int(ptC[0]), int(ptC[1]))
	ptD = (int(ptD[0]), int(ptD[1]))
	ptA = (int(ptA[0]), int(ptA[1]))
    
	# draw the bounding box of the AprilTag detection
	cv2.line(img, ptA, ptB, (0, 255, 0), 2)
	cv2.line(img, ptB, ptC, (0, 255, 0), 2)
	cv2.line(img, ptC, ptD, (0, 255, 0), 2)
	cv2.line(img, ptD, ptA, (0, 255, 0), 2)

	# draw the center (x, y)-coordinates of the AprilTag
	(cX, cY) = (int(r.center[0]), int(r.center[1]))
	cv2.circle(img, (cX, cY), 5, (0, 0, 255), -1)

	# draw the tag family on the image
	tagFamily = r.tag_family.decode("utf-8")
	cv2.putText(img, tagFamily, (ptA[0], ptA[1] - 15),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	print("[INFO] tag family: {}".format(tagFamily))

# show the output image after AprilTag detection
cv2.imshow("Image", img)
cv2.waitKey(0)