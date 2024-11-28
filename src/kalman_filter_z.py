import numpy as np
import cv2
class KalmanFilter2D:
    def __init__(self, x_pos, y_pos, z_pos, dt=1):
        self.dt = dt
        self.F = np.array([[1, dt, 0, 0, 0, 0],  # x_pos
                        [0, 1, 0, 0, 0, 0],  # x_vel
                        [0, 0, 1, dt, 0, 0], # y_pos
                        [0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, dt], # z_pos
                        [0, 0, 0, 0, 0, 1]]) # z_vel
        
        self.H = np.array([[1, 0, 0, 0, 0, 0],  # Measurement matrix
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0]])
        self.R = np.eye(3) * 1000
        self.I = np.eye(6)  # Identity matrix
        self.x = np.array([[x_pos], [0], [y_pos], [0], [z_pos], [0]])  # Initial state
        self.P = np.eye(6) * 1000  # Initial state covariance

    def update(self, measurement):
        y = measurement - np.dot(self.H, self.x)  # Measurement residual
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # Residual covariance
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))  # Kalman gain
        self.x = self.x + np.dot(K, y)  # Updated state
        self.P = np.dot(self.I - np.dot(K, self.H), self.P)  # Updated covariance

    def predict(self, u=None):
        if u is None:
            u = np.zeros((6, 1))  # No external motion by default
        self.x = np.dot(self.F, self.x) + u  # Predicted state
        self.P = np.dot(self.F, np.dot(self.P, self.F.T))  # Predicted covariance
    
class KalmanFilterManager:
    def __init__(self):
        self.kalman_filters = {}

    def initialize_filters(self, measurements):
            for mes in measurements:
                class_id, x, y, z = mes
                if class_id not in self.kalman_filters:
                    self.kalman_filters[class_id] = {}
                obj_id = f"{class_id}_{len(self.kalman_filters[class_id])}"
                kf = KalmanFilter2D(x,y,z)
                self.kalman_filters[class_id][obj_id] = kf

    def update(self, measurements):
        for class_id, filters in self.kalman_filters.items():
            can_continue = False
            for mes in measurements:
                if mes[0] == class_id:
                    can_continue = True
                    break
            if not can_continue:
                continue

            # Get predictions
            predictions = {}
            for obj_id, kf in filters.items():
                kf.predict()
                predictions[obj_id] = kf.x

            # Pair predictions with closest measurements
            obj_locations = []
            for mes in measurements:
                if mes[0] == class_id:
                    x, y, z = mes[1], mes[2], mes[3]
                    obj_locations.append((x, y, z))
            if len(obj_locations) == 0:
                continue

            for obj_id, pred in predictions.items():
                pred_x, pred_y, pred_z = pred[0, 0], pred[2, 0], pred[4, 0]
                if len(obj_locations) == 0: # if we dont find a measurement for a prediction, we dont update the prediction
                    break
                closest_obj = min(obj_locations, key=lambda m: np.linalg.norm((pred_x - m[0], pred_y - m[1], pred_z - m[2])))
                obj_locations.remove(closest_obj)
                mes_for_kf = np.array([[closest_obj[0]], [closest_obj[1]], [closest_obj[2]]])
                kf = filters[obj_id]
                kf.update(mes_for_kf)
                print("updated")

    def predict(self):
        predictions = {}
        for class_id, filters in self.kalman_filters.items():
            predictions[class_id] = {}
            for obj_id, kf in filters.items():
                kf.predict()
                predictions[class_id][obj_id] = np.dot(kf.H, kf.x)
        return predictions

    def get_class_counts(self):
        class_counts = {class_id: len(filters) for class_id, filters in self.kalman_filters.items()}
        return class_counts
    
    def get_predictions_list(self):
        result = []
        predictions = self.predict()
        for class_id, filters in predictions.items():
            for obj_id, state in filters.items():
                (x, y, z) = int(state[0]), int(state[1]), int(state[2])
                w, h = 20, 20  # Assuming width and height are not tracked
                result.append([class_id, x, y, z, w, h])
        return result