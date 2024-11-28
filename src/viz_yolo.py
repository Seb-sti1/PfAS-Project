import cv2
import numpy as np
from kalman_filter_z import KalmanFilterManager
from ultralytics import YOLO
from load import load_stereo_images, load_labels
from detect import detect_objects

# Load YOLO model
model = YOLO("runs/detect/train4/weights/best.pt")
# Initialize Kalman filter


def draw_predictions(image, detections, color, class_counts= None):
    cv2.putText(image, "Kalman Predictions", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    for det in detections:
        class_id, x_center, y_center, z,  w, h = det
        cv2.rectangle(image, (x_center - w // 2, y_center - h // 2), (x_center + w // 2, y_center + h // 2), color, 2)
        cv2.putText(image, model.names[class_id], (x_center - w // 2, y_center - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
        cv2.putText(image, f"Kalman prediction counts:", (image.shape[1] - 270, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        for class_id in class_counts:
            label= "person" if class_id == 0 else "cyclist" if class_id == 1 else "car"
            cv2.putText(image, f"Count {label}: {class_counts[class_id]}", (image.shape[1] - 200, 30 + 20 * class_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

def draw_detections(image, detections, color, class_counts= None):
    cv2.putText(image, "Yolo Detections", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    for det in detections:
        class_id, x_center, y_center,  w, h = det
        cv2.rectangle(image, (x_center - w // 2, y_center - h // 2), (x_center + w // 2, y_center + h // 2), color, 2)
        cv2.putText(image, model.names[class_id], (x_center - w // 2, y_center - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
        cv2.putText(image, f"Yolo detection counts:", (image.shape[1] - 270, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        for class_id in class_counts:
            label= "person" if class_id == 0 else "cyclist" if class_id == 1 else "car"
            cv2.putText(image, f"Count {label}: {class_counts[class_id]}", (image.shape[1] - 200, 110 + 20 * class_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
       

# Main tracking loop
def main(dataset="rec_data", sequence_name="seq_02"):
    labels_pd = load_labels(dataset, sequence_name)
    colors = {"Pedestrian": (255, 255, 255), "Cyclist": (0, 0, 255), "Car": (0, 255, 0)}
    kalman_colors = {"person": (255, 255, 255), "bicycle": (0, 0, 255), "car": (0, 255, 0)}

    # Dictionary to store Kalman filters for each object
    kf_manager = KalmanFilterManager()

    initial_pred= True
    for frame_idx, (left_image, _) in enumerate(load_stereo_images(dataset, sequence_name)):
        frame_height, frame_width = left_image.shape[:2]
        # print(frame_height, frame_width)
        detections = detect_objects(model, left_image)  # YOLO detections
        measurements = [list(det[:3]) + [0] for det in detections]
        # Loop through detected objects
        if initial_pred:
            kf_manager.initialize_filters(measurements)
            initial_pred = False
        else:
            kf_manager.update(measurements)


        class_counts = kf_manager.get_class_counts()
        for class_id in class_counts:
            if class_id == 0:
                print(f"Image: {frame_idx }, Number of people detected by Kalman filter: {class_counts[class_id]}")
            elif class_id == 1:
                print(f"Image: {frame_idx },Number of bicycles detected by Kalman filter: {class_counts[class_id]}")
            elif class_id == 2:
                print(f"Image: {frame_idx },Number of cars detected by Kalman filter: {class_counts[class_id]}")
        
        # Draw Kalman predictions
        predictions = kf_manager.get_predictions_list()
        draw_predictions(left_image, predictions, (0,255,0), class_counts)

        # Draw YOLO detections
        det_class_counts={0:0,1:0,2:0}
        draw_dets = []
        for (class_id,x,y,w,h) in detections:
            det_class_counts[class_id]+=1
            draw_dets.append((class_id,x,y,30,30))
        draw_detections(left_image, draw_dets, (0,0,255),det_class_counts)
              
    
        # Save or display frame
        cv2.imwrite(f"output_2d_viz_xyz/{frame_idx}_left.png", left_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if frame_idx == 100:
            break

if __name__ == "__main__":
    main()
