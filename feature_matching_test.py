import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

from utils import extrinsic_matrix, homography_decomposition

def orb_homography(img1, img2, MIN_MATCH_COUNT=10):

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Create Brute-Force Matcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # If enough matches are found, cmpute homography
    if len(matches)>MIN_MATCH_COUNT:
        # Obtain matched pairs of keypoints
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1, 1, 2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1, 1, 2)
        
        # Calculate homography using RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        return M, mask, kp1, kp2, matches

    else:
        print("Not enough matches are found - %d/%d" % (len(matches),MIN_MATCH_COUNT))
        
        return None, None, None, None, None


if __name__ == "__main__":

    # Marker and camera parameters
    fx, fy, cx, cy = (739.2116337887949, 731.2693931923594, 472.1271812307942, 265.5094352085958)
    K = extrinsic_matrix(fx, fy, cx, cy)

    img1 = cv2.imread('images/room0.jpg',0)
    img2 = cv2.imread('images/room1.jpg',0)

    M, mask, kp1, kp2, matches = orb_homography(img1, img2)
    num, Rs, Ts, Ns = cv2.decomposeHomographyMat(M, K)
    print(M)
    for i in range(num):
        print(Rs[i])
        print(Ts[i])
        print(Ns[i])

    R, t = homography_decomposition(M, K)
    print(R, t)

    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0, 0],[0, h-1],[w-1, h-1],[w-1, 0] ]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    # Draw inliners
    draw_params = dict(matchColor = (0, 255, 0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

    plt.imshow(img3, 'gray'),plt.show()