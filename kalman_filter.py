import numpy as np
from scipy.sparse import coo_matrix, block_diag

class Kalman:
    def __init__(self):

        # State vector [x, y, z, yaw, pitch, roll, vx, vy, vz, vyaw, vpitch, vroll]
        self.x = np.asarray((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        
        # State covariance
        var_init = 1000
        self.P = np.eye(12) * var_init

        # Process error covariance
        self.Q = np.diag((0, 0.01, 0, 0.01, 0, 0.01, 0, 0.0025, 0, 0.0025, 0, 0.0025))
        
        # Sensor measurement covariance
        self.W = np.diag((0.01, 0.01, 0.1, 0.0025, 0.0025, 0.0025))


    def predict_step(self, dt=0):
        A = np.eye(12) + np.eye(12, k = 6) * dt

        self.x = A @ self.x
        self.P = A @ self.P @ np.transpose(A) + self.Q


    def update_step(self, z):
        # Measurement vector [x, y, z, yaw, pitch, roll]
        H = np.eye(6, 12)
        K = self.P @ np.transpose(H) @ np.linalg.inv(H @ self.P @ np.transpose(H) + self.W)

        self.x = self.x + K @ (z - H @ self.x)
        self.P = self.P - K @ H @ self.P