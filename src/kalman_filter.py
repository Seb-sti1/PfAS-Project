import numpy as np
import cv2

class KalmanFilter2D:
    def __init__(self, dt=1/25):
        self.dt = dt
        self.F = np.array([[1, dt, 0.5 * dt**2, 0, 0, 0],  # State transition matrix
                           [0, 1, dt, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 1, dt, 0.5 * dt**2],
                           [0, 0, 0, 0, 1, dt],
                           [0, 0, 0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0, 0, 0],  # Measurement matrix
                           [0, 0, 0, 1, 0, 0]])
        self.R = np.array([[10, 0],  # Measurement noise covariance
                           [0, 10]])
        self.I = np.eye(6)  # Identity matrix

    def initialize(self):
        x = np.zeros((6, 1))  # Initial state [x, vx, ax, y, vy, ay]
        P = np.eye(6) * 1000  # Initial state covariance
        return x, P

    def update(self, x, P, Z):
        y = Z - np.dot(self.H, x)  # Measurement residual
        S = np.dot(self.H, np.dot(P, self.H.T)) + self.R  # Residual covariance
        K = np.dot(P, np.dot(self.H.T, np.linalg.inv(S)))  # Kalman gain
        x = x + np.dot(K, y)  # Updated state
        P = np.dot(self.I - np.dot(K, self.H), P)  # Updated covariance
        return x, P

    def predict(self, x, P, u=None):
        if u is None:
            u = np.zeros((6, 1))  # No external motion by default
        x = np.dot(self.F, x) + u  # Predicted state
        P = np.dot(self.F, np.dot(P, self.F.T))  # Predicted covariance
        return x, P

def is_same_object(kf, x_kf, x, y, vx, vy, class_id, position_threshold=200, velocity_threshold=5):
    pred_x, pred_y = x_kf[0, 0], x_kf[3, 0]
    pred_vx, pred_vy = x_kf[1, 0], x_kf[4, 0]
    
    position_distance = np.sqrt((pred_x - x)**2 + (pred_y - y)**2)
    velocity_distance = np.sqrt((pred_vx - vx)**2 + (pred_vy - vy)**2)
    
    # return position_distance < position_threshold and velocity_distance < velocity_threshold
    return position_distance < position_threshold

def update_kalman_filters(detections, kalman_filters):
    for det in detections:
        class_id, x, y, w, h, vx, vy = det

        # Check if the detection matches any existing object
        matched = False
        for obj_id, kf in kalman_filters.get(class_id, {}).items():
            x_kf, P_kf = kf.predict(kf.x, kf.P)
            if is_same_object(kf, x_kf, x, y, vx, vy, class_id):
                # Update existing Kalman filter
                z_kf = np.array([[x], [y]])
                kf.x, kf.P = kf.update(x_kf, P_kf, z_kf)
                matched = True
                break

        if not matched:
            # Initialize Kalman filter for new objects
            if class_id not in kalman_filters:
                kalman_filters[class_id] = {}
            obj_id = f"{class_id}_{len(kalman_filters[class_id])}"
            kf = KalmanFilter2D()
            kf.x, kf.P = kf.initialize()
            kalman_filters[class_id][obj_id] = kf

            # Update with detection
            z_kf = np.array([[x], [y]])
            kf.x, kf.P = kf.update(kf.x, kf.P, z_kf)

    return kalman_filters

# Example usage:
kf = KalmanFilter2D()
kalman_filters = {}
detections = [
    (0, 100, 200, 50, 100, 5, 5),
    (0, 105, 205, 50, 100, 5, 5),
    (1, 300, 400, 100, 200, 5, 5),
    (1, 305, 405, 100, 200, 5, 5)
]

kalman_filters = update_kalman_filters(detections, kalman_filters)
print(kalman_filters)