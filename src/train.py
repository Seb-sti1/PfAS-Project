import argparse
import cv2
import numpy as np
import os
import pandas as pd
from load import load_labels, load_stereo_images
from ultralytics import YOLO

# Load a COCO-pretrained YOLO model
model = YOLO("yolov8n.pt")

# Define the base data location
data_location = os.path.join(os.path.dirname(__file__), "..")

def prepare_yolo_dataset(data_name, sequence_name):
    labels_pd = load_labels(data_name, sequence_name)
    yolo_images_dir = os.path.join(data_location, data_name, sequence_name, "yolo", "images")
    yolo_labels_dir = os.path.join(data_location, data_name, sequence_name, "yolo", "labels")
    
    os.makedirs(yolo_images_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)
    
    for i, (left_image, _) in enumerate(load_stereo_images(data_name, sequence_name)):
        yolo_image_path = os.path.join(yolo_images_dir, f"{i:06d}.jpg")
        cv2.imwrite(yolo_image_path, left_image)
        
        label_path = os.path.join(yolo_labels_dir, f"{i:06d}.txt")
        with open(label_path, 'w') as f:
            current_labels = labels_pd[labels_pd["frame"] == i]
            for index, row in current_labels.iterrows():
                if row["type"] == "Pedestrian":
                    class_id = 0  # person
                elif row["type"] == "Car":
                    class_id = 2  # car
                elif row["type"] == "Cyclist":
                    class_id = 1  # bicycle
                else:
                    continue  # Skip other types

                x_center = (row["bbox_left"] + row["bbox_right"]) / 2 / left_image.shape[1]
                y_center = (row["bbox_top"] + row["bbox_bottom"]) / 2 / left_image.shape[0]
                width = (row["bbox_right"] - row["bbox_left"]) / left_image.shape[1]
                height = (row["bbox_bottom"] - row["bbox_top"]) / left_image.shape[0]
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def train_model_on_sequence(data_name, sequence_name):
    prepare_yolo_dataset(data_name, sequence_name)
    
    # Create a data.yaml file for YOLO training
    data_yaml = f"""
    train: {os.path.join(data_location, data_name, sequence_name, "yolo", "images")}
    val: {os.path.join(data_location, data_name, sequence_name, "yolo", "images")}
    nc: 3
    names: ['person', 'bicycle', 'car']
    """
    
    with open(os.path.join(data_location, data_name, sequence_name, "yolo", "data.yaml"), 'w') as f:
        f.write(data_yaml)
    
    # Train the model and save the updated model
    model.train(data=os.path.join(data_location, data_name, sequence_name, "yolo", "data.yaml"), epochs=100, imgsz=640, save=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="The dataset", choices=["rec_data"], default="rec_data")
    parser.add_argument("--sequence", help="The sequence", choices=["seq_19"], default="seq_19")
    args = parser.parse_args()

    train_model_on_sequence(args.dataset, args.sequence)