import os.path

import cv2
import pandas


def load_images(seq_path):
    filenames = sorted([filename for filename in os.listdir(os.path.join(seq_path, "image_02", "data"))])
    for name in filenames:
        left_image = cv2.imread(os.path.join(seq_path, "image_02", "data", name))
        right_image = cv2.imread(os.path.join(seq_path, "image_03", "data", name))
        yield left_image, right_image


def add_labels(images, labels, colors):
    for index, row in labels.iterrows():
        top_left = (int(row["bbox_left"]), int(row["bbox_top"]))
        bot_right = (int(row["bbox_right"]), int(row["bbox_bottom"]))

        images = cv2.rectangle(images, top_left, bot_right, colors[row["type"]], 3)


def main():
    raw_path = os.path.join(os.path.dirname(__file__), "..", "raw_data")
    rec_path = os.path.join(os.path.dirname(__file__),"..",  "rec_data")
    sequence_name = "seq_02"

    names = ["frame", "track id", "type", "truncated", "occluded",
             "alpha", "bbox_left", "bbox_top", "bbox_right", "bbox_bottom",
             "height", "width", "length", "x", "y", "z", "rotation_y"]

    labels_pd = pandas.read_csv(os.path.join(rec_path, sequence_name, "labels.txt"),
                                sep=" ",
                                names=names)

    colors = {"Pedestrian": (255, 0, 0),
              "Cyclist": (0, 255, 0),
              "Car": (0, 0, 255)}

    for i, (left_image, right_image) in enumerate(load_images(os.path.join(rec_path, sequence_name))):
        current_labels = labels_pd[labels_pd["frame"] == i]

        add_labels(left_image, current_labels, colors)
        add_labels(right_image, current_labels, colors)

        cv2.imshow("left and right images", left_image)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
