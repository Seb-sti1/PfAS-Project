import os

import cv2
import numpy as np

from viz import load_images


def main():
    raw_path = os.path.join(os.path.dirname(__file__), "..", "raw_data")
    rec_path = os.path.join(os.path.dirname(__file__), "..", "rec_data")
    sequence_name = "seq_02"

    for i, ((raw_left, raw_right), (rec_left, rec_right)) in enumerate(
            zip(load_images(os.path.join(raw_path, sequence_name)),
                load_images(os.path.join(rec_path, sequence_name)))):
        raw = np.concatenate([raw_left, raw_right], axis=1)
        rec = np.concatenate([rec_left, rec_right], axis=1)

        cv2.imshow("raw", cv2.resize(raw, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR))
        cv2.imshow("rec", cv2.resize(rec, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR))
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
