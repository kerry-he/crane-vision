##################################################################
## TRANSFORMS
## Library which contains functions required to calculate 
## homographies from given Euclidian transforms.
##################################################################

# Libraries
import numpy as np
import cv2
import transforms3d

def translation_matrix_2d(t):
    # Create a 3x3 perspective transform matrix corresponding to a 2D translation
    T = np.eye(3)
    T[:2, 2] = t

    return T


def rotation_matrix_2d(angle, centre=(0, 0), translation=(0, 0)):
    # Create a 3x3 perspective transform matrix corresponding to a 2D rotation
    M = np.eye(3)
    c, s = np.cos(angle), np.sin(angle)

    # Insert 2D rotation matrix component
    M[:2, :2] = np.array(((c, -s), (s, c)))

    # Insert centering translation component
    M[:2, 2] = np.array([(1 - c) * centre[0] + s * centre[1],
                         -s * centre[0] + (1 - c) * centre[1]])

    # Insert additional translational component
    M[:2, 2] += np.array([c * translation[0] - s * translation[1],
                          s * translation[0] + c * translation[1]])

    return M


def homography_from_transform(R=np.eye(3), t=np.zeros(3), n=np.zeros(3), d=1.):
    # Calculate Euclidian homography from given 3D transform
    t = np.reshape(t, (3, 1))
    n = np.reshape(n, (1, 3))

    H = R - np.matmul(t, n) / d

    return H


def crane_transform_2d(R, t, K):
    # Calculate homography transform to remove single pendulum swinging displacement
    # on a 2D plane

    # Calculate depth to image plane
    n = np.array([0., 0., 1.])
    depth = np.dot(np.matmul(R, t), n)

    # Find Euler rotation to isolate z rotation to estimate 2D rotation
    euler = transforms3d.euler.mat2euler(R)
    psi = euler[-1]

    # Calculate offset assuming single pendulum model
    l = 200.
    pendulum_offset = np.matmul(
        (np.eye(3) - np.linalg.inv(R)), np.array([0., l, 0.]))

    # Calculate homography transform
    R = rotation_matrix_2d(psi)
    t = translation_matrix_2d(pendulum_offset[:2] / depth)

    # H = K @ R @ t @ inv(K)
    H = np.matmul(K, np.matmul(R, np.matmul(t, np.linalg.inv(K))))

    return H


def crane_transform_3d(R, t, K):
    # Calculate homography transform to remove single pendulum swinging displacement
    # in 3D space

    # Calculate depth to image plane
    n = np.array([0., 0., 1.])
    depth = np.dot(np.matmul(R, t), n)
    depth_after_rotation = (np.matmul(R, np.array([[0, 0, depth]]).T))[2]

    # Calculate offset assuming single pendulum model
    l = 200.
    pendulum_offset = np.matmul(
        (np.linalg.inv(R) - np.eye(3)), np.array([0., l, 0.]))
    pendulum_offset = np.matmul(R, pendulum_offset)

    # Calculate homography transform
    H_rotation = R
    H_translation = homography_from_transform(
        t=pendulum_offset, n=n, d=depth_after_rotation)

    # H = K @ H_translation @ H_rotation @ inv(K)
    H = np.matmul(K, np.matmul(
        H_translation, np.matmul(H_rotation, np.linalg.inv(K))))

    return H


def crane_transform_given(t_boom, t_boom_origin, d_camera_origin, H_camera, H_camera_origin, K):
    # Calculate homography transform to follow movement of boom
    n = np.array([0., 0., 1.])

    # Calculate total displacement of boom from origin position
    t = t_boom - t_boom_origin
    t[2] *= -1

    # Calculate homogaphy transform
    H = homography_from_transform(t=t, n=n, d=d_camera_origin)
    H = np.matmul(K, np.matmul(H, np.linalg.inv(K)))
    H = np.matmul(H, np.matmul(H_camera_origin, np.linalg.inv(H_camera)))

    return H


def crane_transform_depth(R, t, K, min_depth):
    # Calculate homography transform to remain a distance > min_depth away from marker
    n = np.array([0., 0., 1.])
    depth = np.dot(np.matmul(R, t), n)

    # Calculate homography transform
    if depth < min_depth:
        t = np.array([0., 0., depth - min_depth])
        H = homography_from_transform(t=t, n=n, d=depth)
        H = np.matmul(K, np.matmul(H, np.linalg.inv(K)))
    else:
        H = np.eye(3)

    return H
