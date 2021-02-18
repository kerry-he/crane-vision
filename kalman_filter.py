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
        self.Q = np.diag((0., 0., 0., 0., 0., 0., 0.5, 0.5, 0.5, 0.00001, 0.00001, 0.00001))
        
        # Sensor measurement covariance
        self.W = np.diag((0.001, 0.001, 0.015, 4.5e-7, 4.5e-7, 2.5e-9))


    def predict_step(self, dt=0):
        A = np.eye(12) + np.eye(12, k = 6) * dt

        self.x = np.matmul(A, self.x)
        self.P = np.matmul(A, np.matmul(self.P, np.transpose(A))) + self.Q


    def update_step(self, z):
        # Measurement vector [x, y, z, yaw, pitch, roll]
        H = np.eye(6, 12)
        K = np.matmul(self.P, np.matmul(np.transpose(H), np.linalg.inv(np.matmul(H, np.matmul(self.P, np.transpose(H))) + self.W)))

        self.x = self.x + np.matmul(K, (z - np.matmul(H, self.x)))
        self.P = self.P - np.matmul(K, np.matmul(H, self.P))

class KalmanHomography:
    def __init__(self):

        # State vector [H, H_derivative]
        self.x = np.zeros(18)
        
        # State covariance
        var_init = 1000
        self.P = np.eye(18) * var_init

        # Process error covariance
        self.Q = np.diag((0., 0., 0., 0., 0., 0., 0., 0., 0., 0.0005, 0.0005, 0.05, 0.0005, 0.0005, 0.05, 0.0005, 0.0005, 0.0005))
        
        # Sensor measurement covariance
        self.W = np.diag((0.01, 0.01, 1., 0.01, 0.01, 1., 0.01, 0.01, 0.01))


    def predict_step(self, dt=0):
        A = np.eye(18) + np.eye(18, k = 9) * dt

        self.x = np.matmul(A, self.x)
        self.P = np.matmul(A, np.matmul(self.P, np.transpose(A))) + self.Q


    def update_step(self, z):
        # Measurement vector [x, y, z, yaw, pitch, roll]
        H = np.eye(9, 18)
        K = np.matmul(self.P, np.matmul(np.transpose(H), np.linalg.inv(np.matmul(H, np.matmul(self.P, np.transpose(H))) + self.W)))

        self.x = self.x + np.matmul(K, (z - np.matmul(H, self.x)))
        self.P = self.P - np.matmul(K, np.matmul(H, self.P))