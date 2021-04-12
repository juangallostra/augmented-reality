import numpy as np
from abc import ABC, abstractmethod


class KalmanFilter(ABC):
    """
    Generic Kalman Filter implementation
    """
    def __init__(self):
        # Variables to store the current state of the Kalman filter
        self.x = None  # Predicted state
        self.P = None  # Prediced noise
        # self.z = None  # Measurements -- Observations are not required as part of the state
        self._A = None  # Process model
        self._H = None  # Measurement model
        self._Q = None  # Process noise
        self._R = None  # Measurament noise
        self._K = None  # Kalman gain

    def get_current_state(self):
        return self.x

    def get_current_noise(self):
        return self.P

    @abstractmethod
    def get_A(self, **kwargs):
        pass

    @abstractmethod
    def get_H(self, **kwargs):
        pass

    @abstractmethod
    def get_Q(self, **kwargs):
        pass

    @abstractmethod
    def get_R(self, **kwargs):
        pass

    def reset(self):
        """Reset filter"""
        self.__init__()

    def init(self, x, P):
        """Initial system state"""
        self.x = x
        self.P = P

    def __update_models_and_noise(self, **kwargs):
        """ If any of the models has to be updated with some data, do it here"""
        self._A = self.get_A(**kwargs)
        self._H = self.get_H(**kwargs)
        self._Q = self.get_Q(**kwargs)
        self._R = self.get_R(**kwargs)

    def __project_state(self):
        """ x_k = A*x_(k-1) + B*u_k"""
        self.x = np.matmul(self._A, self.x)

    def __project_covariance(self):
        """ P_k = A*P_(k-1)*A' + Q """
        self.P = np.matmul(
            np.matmul(self._A, self.P), 
            np.transpose(self._A)
        ) + self._Q

    def __compute_gain(self):
        """ K_k = P_k*H'*(H*P_K*H' + R)^-1 """
        self.K = np.matmul(
            np.matmul(self.P, np.transpose(self._H)),
            np.linalg.inv(
                np.matmul(
                    self._H,
                    np.matmul(self.P,np.transpose(self._H))
                ) + self._R
            )
        )

    def __correct_state(self, z):
        """ x_k = x_K + K_k*(z_k - H*x_k) """
        self.x = self.x + np.matmul(
            self.K,
            z - np.matmul(self._H, self.x)
        )

    def __correct_covariance(self):
        """ P_k = (I - K_k*H)*P_k """
        self.P = np.matmul(np.eye(16) - np.matmul(self.K, self._H), self.P)

    def predict(self, **kwargs):
        """ Prediction step """
        self.__update_models_and_noise(**kwargs)
        self.__project_state()
        self.__project_covariance()

    def correct(self, measurements):
        """ Correction step """
        self.__compute_gain()
        self.__correct_state(measurements)
        self.__correct_covariance()


class KalmanTracker(KalmanFilter):
    """
        As a first test, track the four corners of the reference surface.
        This means that we have 16 states (2 pos, 2 vels for each corner)
        Dimensions are then:
        x -> 16 x 1 (States - 4 corners (u, v) and its velocities (u', v') => 16 values (8 pos + 8 vel))
        z -> 8 x 1 (Observations - 4 corners => 8 values)
        A -> 16 x 16 (State transition model)
        H -> 8 x 16 (Observation model)
        P -> 16 x 16 (Error covariance)
        Q -> 16 x 16 (Process noise covariance)
        R -> 8 x 8 (Observation noise covariance)
        K -> 16 x 8 (Kalman Gain)
    """
    def __init__(self):
        super().__init__()

    def get_A(self, dt=None):
        return np.eye(16) + np.diag(np.ones(8), 8)*dt  # 16x16

    def get_H(self, **kwargs):
        return np.concatenate((np.eye(8), np.zeros([8, 8])), 1)  # 8x16

    def get_Q(self, q=0.5, **kwargs):
        a = np.eye(8)*q**2
        b = np.zeros([8, 8])
        return np.block([[b, b],[b, a]])  # 16x16

    def get_R(self, r=0.5, **kwargs):
        return np.eye(8) * r**2  # 8x8
