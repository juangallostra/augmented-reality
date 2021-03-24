import numpy as np
from dataclasses import dataclass

@dataclass
class KalmanSettings:
    """Class for keeping the initial values of the system"""
    A: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    R: np.ndarray

class KalmanTracker():
    """
    As a first test, track the four corners of the reference surface.
    This means that we have 16 states (2 pos, 2 vels for each corner)
    Dimensions are then:
    x -> 16 x 1 (States)
    z -> 8 x 1 (Observations)
    A -> 16 x 16 (State transition model)
    H -> 8 x 16 (Observation model)
    P -> 16 x 16 (Error covariance)
    Q -> 16 x 16 (Process noise covariance)
    R -> 8 x 8 (Observation noise covariance)
    K -> 16 x 8 (Kalman Gain)
    """

    def __init__(self, A, H, Q, R):
        # Variables to store current state of the Kalman filter
        self.x = None  # State
        self.P = None  # Prediced noise
        # self.z = None  # Measurements -- Observations are not required as part of the state
        self.A = A  # Process model
        self.H = H  # Measurement model
        self.Q = Q  # Process noise
        self.R = R  # Measurament noise
        self.K = None  # Kalman gain
        self.original_state = KalmanSettings(A, H, Q, R)

    def __load_original_settings(self):
        self.A = self.original_state.A
        self.H = self.original_state.H
        self.Q = self.original_state.Q
        self.R = self.original_state.R

    def reset(self, x, P):
        """Reset filter"""
        self.init(x, P)
        self._load_original_settings()

    def init(self, x, P):
        """Initial system state"""
        self.x = x
        self.P = P

    def __update_models(self, deltat):
        """ If any of the models has to be updated from some data, do it here"""

    def __project_state(self):
        """ x_k = A*x_(k-1) + B*u_k"""
        self.x = np.matmul(self.A, self.x)

    def __project_covariance(self):
        """ P_k = A*P_(k-1)*A' + Q """
        self.P = np.matmul(
            np.matmul(self.A, self.P),
            np.transpose(self.A)
        ) + self.Q

    def __compute_gain(self):
        """ K_k = P_k*H'*(H*P_K*H' + R)^-1 """
        self.K = np.matmul(
            np.matmul(
                self.P,
                np.transpose(self.H)
            ),
            np.inv(
                np.matmul(
                    self.H,
                    np.matmul(
                        self.P,
                        np.transpose(self.H)
                    )
                ) + self.R
            )
        )

    def __correct_state(self, z):
        """ x_k = x_K + K_k*(z_k - H*x_k) """
        self.x = self.x + np.matmul(
            self.K,
            z - np.matmul(self.H, self.x)
        )

    def __correct_covariance(self):
        """ P_k = (I - K_k*H)*P_k """
        self.P = np.matmul(np.eye(16) - np.matmul(self.K, self.H), self.P)

    def predict(self, deltat):
        """ Prediction step """
        self.__update_models(deltat)
        self.__project_state()
        self.__project_covariance()

    def correct(self, measurements):
        """ Correction step """
        self.__compute_gain()
        self.__correct_state(measurements)
        self.__correct_covariance()
