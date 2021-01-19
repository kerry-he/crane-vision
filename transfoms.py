import numpy as np
import time
import cv2
from scipy.spatial.transform import Rotation


def transformation_matrix(R, t):
    R = np.asarray(R)
    t = np.reshape(t, 3)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

def translation_matrix_2d(t):
    T = np.eye(3)
    T[:2, 2] = t

    return T


def rotation_matrix_2d(angle, centre=(0, 0), translation=(0, 0)):
    M = np.eye(3)
    c, s = np.cos(angle), np.sin(angle)

    M[:2, :2] = np.array(((c, -s), (s, c)))

    M[:2, 2] = np.array([(1 - c) * centre[0] + s * centre[1],
                  -s * centre[0] + (1 - c) * centre[1]])

    M[:2, 2] += np.array([c * translation[0] - s * translation[1],
                  s * translation[0] + c * translation[1]])

    return M


def rigid_from_transform(R, t, K):
    # Find Euler rotation to isolate z rotation
    r = Rotation.from_matrix(R)
    euler = r.as_euler('xyz')

    # Calculate translation required to always centre the marker
    n = np.array([0, 0, 1])
    depth = np.dot(t, n)
    l = 400
    pendulum_offset = np.array([l * np.sin(euler[-1]), l * (1 - np.cos(euler[-1]))]) / depth

    R = rotation_matrix_2d(euler[-1])
    t = translation_matrix_2d(pendulum_offset)

    M = K @ R @ t @ np.linalg.inv(K)

    return M


def homography_from_transform(R, t, K):

    t = np.reshape(t, (3, 1))
    d = np.array([[0, 0, 1]]) / t[-1]
    H_euclidian = R + t @ d

    H = K @ H_euclidian @ np.linalg.inv(K)

    return H


def homography_from_rotation(R, t, K):

    H = K @ R @ np.linalg.inv(K)

    return H

def filter_swinging(img, R, t, K):

    H = rigid_from_transform(R, t, K)
    img = cv2.warpPerspective(img, H, dsize=img.shape[1::-1])

    return img
