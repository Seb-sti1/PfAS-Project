import os
from typing import Iterator, Tuple

import cv2
import pandas
import pandas as pd
from numpy import ndarray

data_location = os.path.join(os.path.dirname(__file__), "..")


def load_stereo_images(data_name: str, sequence_name: str) -> Iterator[Tuple[ndarray, ndarray]]:
    """
    Load the left and right images for a given dataset (raw_data or rec_data) and sequence (calib, seq_01, seq_02, seq_03)
    :param data_name: either raw_data or rec_data
    :param sequence_name: either calib, seq_01, seq_02, seq_03
    :return: an iterator on the left and right images
    """
    seq_path = os.path.join(data_location, data_name, sequence_name)
    filenames = sorted([filename for filename in os.listdir(os.path.join(seq_path, "image_02", "data"))])
    for name in filenames:
        left_image = cv2.imread(os.path.join(seq_path, "image_02", "data", name))
        right_image = cv2.imread(os.path.join(seq_path, "image_03", "data", name))
        yield left_image, right_image


def load_labels(data_name: str, sequence_name: str) -> pd.DataFrame:
    """
    Load the labels.txt file for a given dataset (raw_data or rec_data) and sequence (seq_01, seq_02)
    :param data_name: either raw_data or rec_data
    :param sequence_name: either seq_01, seq_02
    :return: an iterator on the left and right images
    """
    names = ["frame", "track id", "type", "truncated", "occluded",
             "alpha", "bbox_left", "bbox_top", "bbox_right", "bbox_bottom",
             "height", "width", "length", "x", "y", "z", "rotation_y"]

    return pandas.read_csv(os.path.join(data_location, data_name, sequence_name, "labels.txt"),
                           sep=" ",
                           names=names)