import argparse
import cv2
import numpy as np
from load import load_stereo_images, load_labels
from ultralytics import YOLO

# Load a COCO-pretrained YOLO model
# model = YOLO("yolov8n.pt")
model = YOLO("runs/detect/train4/weights/best.pt")
def draw_true_labels(image, labels, colors):
    for index, row in labels.iterrows():
        top_left = (int(row["bbox_left"]), int(row["bbox_top"]))
        bot_right = (int(row["bbox_right"]), int(row["bbox_bottom"]))
        image = cv2.rectangle(image, top_left, bot_right, colors[row["type"]], 2)
        cv2.putText(image, str(row["type"]), (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[row["type"]], 1)

def draw_predicted_labels(image, labels, colors):
    for class_id, box in labels:
        label = model.names[class_id]
        color = colors[label]
        x, y, w, h = box
        draw_dashed_rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def detect_objects(image):
    results = model(image)
    class_ids = []
    boxes = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if model.names[class_id] in ["person", "bicycle", "car"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert tensor to list first
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                class_ids.append(class_id)
    return [(class_ids[i], boxes[i]) for i in range(len(boxes))]

def draw_dashed_rectangle(image, top_left, bot_right, color, thickness=2):
    x1, y1 = top_left
    x2, y2 = bot_right
    line_type = cv2.LINE_AA
    dash_length = 5

    for i in range(x1, x2, dash_length * 2):
        cv2.line(image, (i, y1), (min(i + dash_length, x2), y1), color, thickness, line_type)
        cv2.line(image, (i, y2), (min(i + dash_length, x2), y2), color, thickness, line_type)
    for i in range(y1, y2, dash_length * 2):
        cv2.line(image, (x1, i), (x1, min(i + dash_length, y2)), color, thickness, line_type)
        cv2.line(image, (x2, i), (x2, min(i + dash_length, y2)), color, thickness, line_type)

def main(dataset= "rec_data", sequence_name="seq_02"):
    labels_pd = load_labels(dataset, sequence_name)
    colors = {"Pedestrian": (255, 255, 255), "Cyclist": (0, 0, 255), "Car": (0, 255, 0)}
    prediction_colors = {"person": (255, 255, 255), "bicycle": (0, 0, 255), "car": (0, 255, 0)}
    for i, (left_image, right_image, name) in enumerate(load_stereo_images(dataset, sequence_name)):
        detections = detect_objects(left_image)
        draw_predicted_labels(left_image, detections, prediction_colors)
        
        current_labels = labels_pd[labels_pd["frame"] == i]
        draw_true_labels(left_image, current_labels, colors)
        
        cv2.imwrite(f"output/{name}_left.png", left_image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        if i == 10:
            break

if __name__ == "__main__":
    main()