import numpy as np

class KalmanTracker():
    """
    As a first test, track the four corners of the reference surface.
    """

    def __init__(self):
        # Variables to store state of the Kalman filter
        self.x = None # State
        self.z = None # Measurements
        self.A = None # Process model
        self.H = None # Measurement model
        self.P = None # Prediced noise
        self.Q = None # Process noise
        self.R = None # Measurament noise
        self.K = None # Kalman gain
        pass

    def reset(self):
        pass

    def __project_state(self):
        """ x_k = A*x_(k-1) + B*u_k"""
        pass

    def __project_covariance(self):
        """ P_k = A*P_(k-1)*A' + Q """
        pass

    def __compute_gain(self):
        """ K_k = P_k*H'*(H*P_K*H' + R)^-1 """
        pass

    def __correct_state(self):
        """ x_k = x_K + K_k*(z_k - H*x_k) """
        pass

    def __correct_covariance(self):
        """ P_k = (I - K_k*H)*P_k """
        pass

    def predict(self):
        """ Prediction step """
        self.__project_state()
        self.__project_covariance()

    def correct(self):
        """ Correction step """
        self.__compute_gain()
        self.__correct_state()
        self.__correct_covariance()
