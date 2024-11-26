import os
from typing import Iterator, Tuple, Optional

import cv2
import numpy as np
import pandas
import pandas as pd
from numpy import ndarray

data_location = os.path.join(os.path.dirname(__file__), "..")


def load_stereo_images(data_name: str, sequence_name: str) -> Iterator[Tuple[Optional[ndarray], Optional[ndarray]]]:
    """
    Load the left and right images for a given dataset (raw_data or rec_data) and sequence (calib, seq_01, seq_02, seq_03)
    :param data_name: either raw_data or rec_data
    :param sequence_name: either calib, seq_01, seq_02, seq_03
    :return: an iterator on the left and right images
    """
    seq_path = os.path.join(data_location, data_name, sequence_name)
    filenames = sorted([filename for filename in os.listdir(os.path.join(seq_path, "image_02", "data"))])
    for name in filenames:
        left_path = os.path.join(seq_path, "image_02", "data", name)
        right_path = os.path.join(seq_path, "image_03", "data", name)
        left_image = None
        right_image = None
        if os.path.exists(left_path):
            left_image = cv2.imread(left_path)
        if os.path.exists(right_path):
            right_image = cv2.imread(right_path)
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


def load_calib_matrix(matrix: str, shape: Tuple[int, ...], image: str = "00") -> np.ndarray:
    """
    :param matrix: eg S, S_rect
    :param shape: the shape of the matrix
    :param image: eg 00
    :return:
    """
    with open(os.path.join(data_location, "rec_data", "calib_cam_to_cam.txt"), 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith(f"{matrix}_{image}"):
                matrix_values = list(map(float, line.split(":")[1].strip().split(" ")))
                return np.array(matrix_values).reshape(shape)
