import argparse

import cv2
import numpy as np

from load import load_stereo_images, load_labels


def add_labels(images, labels, colors):
    for index, row in labels.iterrows():
        top_left = (int(row["bbox_left"]), int(row["bbox_top"]))
        bot_right = (int(row["bbox_right"]), int(row["bbox_bottom"]))

        images = cv2.rectangle(images, top_left, bot_right, colors[row["type"]], 3)


def true_labels(dataset: str, sequence_name: str):
    labels_pd = load_labels(dataset, sequence_name)

    colors = {"Pedestrian": (255, 0, 0),
              "Cyclist": (0, 255, 0),
              "Car": (0, 0, 255)}
    for i, (left_image, right_image) in enumerate(load_stereo_images(dataset, sequence_name)):
        current_labels = labels_pd[labels_pd["frame"] == i]

        add_labels(left_image, current_labels, colors)
        add_labels(right_image, current_labels, colors)

        cv2.imshow("left and right images", left_image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


def raw_and_rec(sequence_name):
    for i, ((raw_left, raw_right), (rec_left, rec_right)) in enumerate(
            zip(load_stereo_images("raw_data", sequence_name),
                load_stereo_images("rec_data", sequence_name))):
        raw = np.concatenate([raw_left, raw_right], axis=1)
        rec = np.concatenate([rec_left, rec_right], axis=1)

        cv2.imshow("raw", cv2.resize(raw, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR))
        cv2.imshow("rec", cv2.resize(rec, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR))
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="The dataset",
                        choices=["rec_data", "rec_data"], default="rec_data")
    parser.add_argument("--sequence", help="The sequence",
                        choices=["calib", "seq_01", "seq_02", "seq_03"], default="seq_01")
    parser.add_argument("type", help="Choose what to show",
                        choices=["trueLabels", "rawAndRec"])
    args = parser.parse_args()

    if args.type == "trueLabels":
        print(f"You're looking at {args.dataset} sequence {args.sequence}.\n"
              f"\t - Press space to see the next.\n"
              f"\t - Press q to quit.\n")
        true_labels(args.dataset, args.sequence)
    elif args.type == "rawAndRec":
        print(f"You're looking at sequence {args.sequence}.\n"
              f"\t - Press space to see the next.\n"
              f"\t - Press q to quit.\n")
        raw_and_rec(args.sequence)


if __name__ == "__main__":
    main()
