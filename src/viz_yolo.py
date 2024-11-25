import cv2
import numpy as np
from kalman_filter_z import KalmanFilterManager
from ultralytics import YOLO
from load import load_stereo_images, load_labels
from detect import detect_objects

# Load YOLO model
model = YOLO("runs/detect/train4/weights/best.pt")
# Initialize Kalman filter


def draw_detections(image, detections, colors):
                            for det in detections:
                                class_id, x_center, y_center, z,  w, h = det
                                color = colors.get(class_id, (0, 255, 0))
                                cv2.rectangle(image, (x_center - w // 2, y_center - h // 2), (x_center + w // 2, y_center + h // 2), color, 2)
                                cv2.putText(image, model.names[class_id], (x_center - w // 2, y_center - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# Main tracking loop
def main(dataset="rec_data", sequence_name="seq_02"):
    labels_pd = load_labels(dataset, sequence_name)
    colors = {"Pedestrian": (255, 255, 255), "Cyclist": (0, 0, 255), "Car": (0, 255, 0)}
    kalman_colors = {"person": (255, 255, 255), "bicycle": (0, 0, 255), "car": (0, 255, 0)}

    # Dictionary to store Kalman filters for each object
    kf_manager = KalmanFilterManager()

    initial_pred= True
    for frame_idx, (left_image, _) in enumerate(load_stereo_images(dataset, sequence_name)):
        detections = detect_objects(model, left_image)  # YOLO detections
        detections = [list(det[:3]) + [0] for det in detections]
        # Loop through detected objects
        if initial_pred:
            kf_manager.initialize_filters(detections)
            initial_pred = False
        else:
            kf_manager.update(detections)


        class_counts = kf_manager.get_class_counts()
        for class_id in class_counts:
            if class_id == 0:
                print(f"Image: {frame_idx }, Number of people detected by Kalman filter: {class_counts[class_id]}")
            elif class_id == 1:
                print(f"Image: {frame_idx },Number of bicycles detected by Kalman filter: {class_counts[class_id]}")
            elif class_id == 2:
                print(f"Image: {frame_idx },Number of cars detected by Kalman filter: {class_counts[class_id]}")
            # Draw YOLO detections
        predictions = kf_manager.get_predictions_list()
        draw_detections(left_image, predictions, kalman_colors)
              
    
        # Save or display frame
        cv2.imwrite(f"output_2d_viz_xyz/{frame_idx}_left.png", left_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_idx == 100:
            break

if __name__ == "__main__":
    main()
