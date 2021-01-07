import numpy as np


def extrinsic_matrix(fx, fy, cx, cy):
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    return K

def homography_decomposition(H, K):
    # Converts a homography H into rotation and pose
    H = H.T
    h1 = H[0]
    h2 = H[1]
    h3 = H[2]
    K_inv = np.linalg.inv(K)
    L = 1 / np.linalg.norm(np.dot(K_inv, h1))
    r1 = L * np.dot(K_inv, h1)
    r2 = L * np.dot(K_inv, h2)
    r3 = np.cross(r1, r2)

    t = L * (K_inv @ h3.reshape(3, 1))
    R = np.array([[r1], [r2], [r3]])
    R = np.reshape(R, (3, 3))

    return R, t

def calculate_homography(src, dst):
    x1, y1 = src[0]
    x2, y2 = src[1]
    x3, y3 = src[2]
    x4, y4 = src[3]

    x1_d, y1_d = dst[0]
    x2_d, y2_d = dst[1]
    x3_d, y3_d = dst[2]
    x4_d, y4_d = dst[3]

    A = np.array([
        [x1, y1, 1, 0, 0, 0, -x1_d*x1, -x1_d*y1, -x1_d],
        [0, 0, 0, x1, y1, 1, -y1_d*x1, -y1_d*y1, -y1_d],
        [x2, y2, 1, 0, 0, 0, -x2_d*x2, -x2_d*y2, -x2_d],
        [0, 0, 0, x2, y2, 1, -y2_d*x2, -y2_d*y2, -y2_d],
        [x3, y3, 1, 0, 0, 0, -x3_d*x3, -x3_d*y3, -x3_d],
        [0, 0, 0, x3, y3, 1, -y3_d*x3, -y3_d*y3, -y3_d],
        [x4, y4, 1, 0, 0, 0, -x4_d*x4, -x4_d*y4, -x4_d],
        [0, 0, 0, x4, y4, 1, -y4_d*x4, -y4_d*y4, -y4_d],
    ])

    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape((3, 3))

    return H

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

        return M

    else:
        print("Not enough matches are found - %d/%d" % (len(matches),MIN_MATCH_COUNT))
        
        return None