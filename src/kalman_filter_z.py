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
    def __init__(self, frame_width= 370, frame_height= 1224):
        self.kalman_filters = {}
        self.frame_height = frame_height
        self.frame_width = frame_width

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
            # Get predictions for the current class
            predictions = {}
            for obj_id, kf in filters.items():
                kf.predict()
                predictions[obj_id] = kf.x

            # Get measurements for the current class
            mes_obj_locations = self.get_mes_obj_locations(measurements, class_id)


            # Remove filters with predictions out of frame
            filters_to_remove = self.get_out_of_frame_predictions(predictions)
            if filters_to_remove is not None:
                for obj_id in filters_to_remove:
                    predictions.pop(obj_id)
                    del filters[obj_id]
          
            # Update filters with measurements closest to them and within a threshold
            for obj_id, pred in predictions.items():
                if len(mes_obj_locations) == 0:
                    break
                mes_for_kf = self.find_closest_measurement(pred, mes_obj_locations)
                if mes_for_kf is not None:
                    kf = filters[obj_id]
                    kf.update(mes_for_kf)
                    print("updated")
                else:
                    print("not updated")

            # Initialize new filters for those measurements that are close to the frame boundary
            if mes_obj_locations:
                filter_input=[]
                for (x, y, z) in mes_obj_locations:
                    if x < 20 or x > self.frame_width - 20 or y < 20 or y > self.frame_height - 20:
                        filter_input.append([class_id, x, y, z])
                self.initialize_filters(filter_input)

    def predict(self):
        predictions = {}
        for class_id, filters in self.kalman_filters.items():
            predictions[class_id] = {}
            for obj_id, kf in filters.items():
                kf.predict()
                predictions[class_id][obj_id] = np.dot(kf.H, kf.x)
        return predictions
    
    def get_out_of_frame_predictions(self, predictions):
        out_of_frame_filters = []
        treshold = 0
        for obj_id, pred in predictions.items():
            x, y, z = pred[0, 0], pred[2, 0], pred[4, 0]
            if x < treshold or x > self.frame_height-treshold or y < treshold or y > self.frame_width - treshold:
                out_of_frame_filters.append(obj_id)
        return out_of_frame_filters

    def find_closest_measurement(self, pred, mes_obj_locations):
        pred_x, pred_y, pred_z = pred[0, 0], pred[2, 0], pred[4, 0]
        closest_mes_obj = min(mes_obj_locations, key=lambda m: np.linalg.norm((pred_x - m[0], pred_y - m[1], pred_z - m[2])))
        distance = np.linalg.norm((pred_x - closest_mes_obj[0], pred_y - closest_mes_obj[1], pred_z - closest_mes_obj[2]))
        threshold = 50  # Define your threshold value here
        if distance > threshold:
            return None  
        mes_obj_locations.remove(closest_mes_obj)
        mes_for_kf = np.array([[closest_mes_obj[0]], [closest_mes_obj[1]], [closest_mes_obj[2]]])
        return mes_for_kf

    def get_mes_obj_locations(self, measurements, class_id):
        mes_obj_locations = []
        for mes in measurements:
            if mes[0] == class_id:
                x, y, z = mes[1], mes[2], mes[3]
                mes_obj_locations.append((x, y, z))
        return mes_obj_locations
    
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