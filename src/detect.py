# Detection function
def detect_objects(model,image):
    results = model(image)
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if model.names[class_id] in ["person", "bicycle", "car"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bbox_center_x = (x1 + x2) // 2
                bbox_center_y = (y1 + y2) // 2
                detections.append((class_id, bbox_center_x, bbox_center_y, x2 - x1, y2 - y1))
    return detections