import cv2
import numpy as np
from kalman_filter import KalmanFilter2D, update_kalman_filters
from ultralytics import YOLO
from load import load_stereo_images, load_labels
from detect import detect_objects

# Load YOLO model
model = YOLO("runs/detect/train4/weights/best.pt")
# Initialize Kalman filter
kf = KalmanFilter2D()

def draw_detections(image, detections, colors):
                            for det in detections:
                                class_id, x, y, w, h = det
                                color = colors.get(class_id, (0, 255, 0))
                                cv2.rectangle(image, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)
                                cv2.putText(image, model.names[class_id], (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# Main tracking loop
def main(dataset="rec_data", sequence_name="seq_02"):
    labels_pd = load_labels(dataset, sequence_name)
    colors = {"Pedestrian": (255, 255, 255), "Cyclist": (0, 0, 255), "Car": (0, 255, 0)}
    kalman_colors = {"person": (255, 255, 255), "bicycle": (0, 0, 255), "car": (0, 255, 0)}

    # Dictionary to store Kalman filters for each object
    kalman_filters = {}
    previous_detections = []

    for frame_idx, (left_image, _, name) in enumerate(load_stereo_images(dataset, sequence_name)):
        detections = detect_objects(model, left_image)  # YOLO detections
        # Loop through detected objects
        for det in detections:
            class_id, x, y, w, h = det
            vx, vy = 0, 0  # Initialize velocities (you may need to calculate these based on previous frames)
            x=x+w//2
            y=y+h//2
            # Update Kalman filters with the new detections
            kalman_filters = update_kalman_filters([(class_id, x, y, w, h, vx, vy)], kalman_filters)
    
            # Get updated position from Kalman filter
            for obj_id, kalman_filter in kalman_filters[class_id].items():
                x_kf, P_kf = kalman_filter.x, kalman_filter.P
                pred_x, pred_y = int(x_kf[0, 0]-w//2), int(x_kf[3, 0]-h//2)
                if class_id == 0:
                    print(f"Image: {frame_idx }, Number of people detected by Kalman filter: {len(kalman_filters[class_id])}")
                elif class_id == 1:
                    print(f"Image: {frame_idx },Number of bicycles detected by Kalman filter: {len(kalman_filters[class_id])}")
                elif class_id == 2:
                    print(f"Image: {frame_idx },Number of cars detected by Kalman filter: {len(kalman_filters[class_id])}")
                # Draw the predicted position
                draw_detections(left_image, [(class_id, pred_x, pred_y, w, h)], kalman_colors)
    
        # Save or display frame
        cv2.imwrite(f"output2/{name}_left.png", left_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_idx == 100:
            break

        # Save or display frame
        cv2.imwrite(f"output2/{name}_left.png", left_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_idx == 100:
            break

if __name__ == "__main__":
    main()
