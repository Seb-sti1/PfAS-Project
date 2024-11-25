import numpy as np
import cv2
with_prev=True
class KalmanFilter2D:
    def __init__(self, dt=1):
        self.dt = dt
        if with_prev:
            self.F = np.array([[1, dt, 0, 0, 0, 0],  # x_pos
                            [1, 0, 0, 0, -1, 0],  # x_vel
                            [0, 0, 1, dt, 0 , 0], # y_pos
                            [0, 0, 1, 0, 0, -1],  # y_vel
                            [1, 0, 0, 0, 0, 0],  # x_prev_pos
                            [0, 0, 1, 0, 0, 0]]) # y_prev_pos
            self.H = np.array([[1, 0, 0, 0, 0, 0],  # Measurement matrix
                            [0, 0, 1, 0, 0, 0]])
            self.R = np.eye(2) * 1000
            self.I = np.eye(6)  # Identity matrix
        else:
            self.F = np.array([[1, dt, 0, 0],  # x_pos
                            [0, 1, 0, 0],  # x_vel
                            [0, 0, 1, dt], # y_pos
                            [0, 0, 0, 0]])  # y_vel
            self.H = np.array([[1, 0, 0, 0],  # Measurement matrix
                            [0, 0, 1, 0]])
            self.R = np.eye(2) * 1000
            self.I = np.eye(4)  # Identity matrix

    def initialize(self, x_pos = 0, y_pos = 0):
        if with_prev:
            x = np.zeros((6, 1))  # Initial state [x, vx, y, vy, x_prev, y_prev]
            x = np.array([[x_pos], [0], [y_pos], [0], [x_pos], [y_pos]])  # Initial state
            P = np.eye(6) * 1000 # Initial state covariance
            return x, P
        else:
            x = np.zeros((4, 1))  # Initial state [x, vx, y, vy, x_prev, y_prev]
            x = np.array([[x_pos], [0], [y_pos], [0]])  # Initial state
            P = np.eye(4) * 1000  # Initial state covariance
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
            if with_prev:
                u = np.zeros((6, 1))
            else:
                u = np.zeros((4, 1))  # No external motion by default
        x = np.dot(self.F, x) + u  # Predicted state
        P = np.dot(self.F, np.dot(P, self.F.T))  # Predicted covariance
        return x, P
    
class KalmanFilterManager:
    def __init__(self):
        self.kalman_filters = {}

    def initialize_filters(self, detections):
            for det in detections:
                class_id, x, y, w, h = det
                if class_id not in self.kalman_filters:
                    self.kalman_filters[class_id] = {}
                obj_id = f"{class_id}_{len(self.kalman_filters[class_id])}"
                kf = KalmanFilter2D()
                kf.x, kf.P = kf.initialize(x, y)
                self.kalman_filters[class_id][obj_id] = kf

    def update(self, detections):
        for class_id, filters in self.kalman_filters.items():
            can_continue = False
            for detection in detections:
                if detection[0] == class_id:
                    can_continue = True
                    break
            if not can_continue:
                continue

            # Get predictions
            predictions = {}
            for obj_id, kf in filters.items():
                kf.x, kf.P = kf.predict(kf.x, kf.P)
                predictions[obj_id] = kf.x

            # Pair predictions with closest measurements
            measurements = []
            for det in detections:
                if det[0] == class_id:
                    x, y = det[1], det[2]
                    measurements.append((x, y))
            if len(measurements) == 0:
                continue

            for obj_id, pred in predictions.items():
                pred_x, pred_y = pred[0, 0], pred[2, 0]
                if len(measurements) == 0: # if we dont find a measurement for a prediction, we dont update the prediction
                    break
                closest_measurement = min(measurements, key=lambda m: np.linalg.norm((pred_x - m[0], pred_y - m[1])))
                measurements.remove(closest_measurement)
                z_kf = np.array([[closest_measurement[0]], [closest_measurement[1]]])
                kf = filters[obj_id]
                kf.x, kf.P = kf.update(kf.x, kf.P, z_kf)
                print("updated")

    def predict(self):
        predictions = {}
        for class_id, filters in self.kalman_filters.items():
            predictions[class_id] = {}
            for obj_id, kf in filters.items():
                kf.x, kf.P = kf.predict(kf.x, kf.P)
                predictions[class_id][obj_id] = kf.x
        return predictions

    def get_class_counts(self):
        class_counts = {class_id: len(filters) for class_id, filters in self.kalman_filters.items()}
        return class_counts
    
    def get_predictions_list(self):
        result = []
        predictions = self.predict()
        for class_id, filters in predictions.items():
            for obj_id, state in filters.items():
                x, y = int(state[0, 0]), int(state[2, 0])
                w, h = 20, 20  # Assuming width and height are not tracked
                result.append([class_id, x, y, w, h])
        return result