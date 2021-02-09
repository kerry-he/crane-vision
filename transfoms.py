import numpy as np
import time
import cv2
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation


def transformation_matrix(R=np.eye(3), t=np.zeros(3)):
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
    l = 200

    # pendulum_offset = np.array([l * np.sin(-euler[-1]), l * (1 - np.cos(-euler[-1]))]) / depth
    # pendulum_offset = np.array([200., 0., 0.])
    pendulum_offset = (np.eye(3) - np.linalg.inv(R)) @ np.array([0., l, 0.]) / depth
    # pendulum_offset = np.linalg.inv(R) @ pendulum_offset
    # print(pendulum_offset)

    # print(-t[:2] / depth)
    # print("")

    R = rotation_matrix_2d(euler[-1])
    t = translation_matrix_2d(pendulum_offset[:2])
    

    # M = np.matmul(K, np.matmul(R, np.matmul(t, np.linalg.inv(K))))
    M = K @ R @ t @ np.linalg.inv(K)

    return M


# def homography_from_transform(R, t, K):

#     t = np.reshape(t, (3, 1))
#     d = np.array([[0, 0, 1]]) / t[-1]
#     H_euclidian = R + t @ d

#     H = K @ H_euclidian @ np.linalg.inv(K)

#     return H


# def homography_from_rotation(R, t, K):

#     H = K @ R @ np.linalg.inv(K)

#     return H


def complete_transform(R, t, K):
    n = np.array([0, 0, 1])
    depth = np.dot(t, n)

    l = 200.
    pendulum_offset = -(np.eye(3) - np.linalg.inv(R)) @ np.array([0., l, 0.])

    # target_depth = depth + pendulum_offset[2]

    # # R = Rotation.from_euler('xyz', [i, 0, 0], degrees=True).as_matrix()
    # offset = -(np.eye(3) - R) @ np.array([[0, 0, target_depth]]).T

    # R = transformation_matrix(R=R)
    # # t = transformation_matrix(t=[0, 0, depth])

    # T = R

    # R = T[:3, :3]
    # t_temp = -T[:3, 3]
    depth2 = (R @ np.array([[0, 0, depth]]).T)[2]
    # t = R @ t
    # t[2] -= 1000
    # print(t)
    t_rot = R @ t
    t_rot[2] -= 1000
    H = R #- np.reshape(-t, (3, 1)) @ np.array([[0., 0., 1.]]) / depth
    H2 = np.eye(3) - np.reshape((R @ pendulum_offset), (3, 1)) @ np.array([[0., 0., 1.]]) / depth

    
    # # print(depth2)
    

    # # offset_from_center = np.reshape(t, (1, 3))
    # # offset_from_center[0, 2] = -target_depth + offset_from_center[0, 2] 

    # H2 = np.eye(3) - np.reshape(-t, (3, 1)) @ np.array([[0., 0., 1.]]) / depth2
    # print(t)
    # H3 = np.eye(3) - np.reshape(pendulum_offset, (3, 1)) @ np.array([[0., 0., 1.]]) / depth

    H = K @ H2 @ H @ np.linalg.inv(K)

    return H


def filter_swinging(img, R, t, K):

    H = complete_transform(R, t, K)
    img = cv2.warpPerspective(img, H, dsize=img.shape[1::-1])

    return img, H
