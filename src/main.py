import argparse

import cv2
import numpy as np
from ultralytics import YOLO

from depth import init_stereo, get_disparity, baseline, R_rect, get_roi_xyz
from kalman_filter import KalmanFilterManager
from load import load_stereo_images
from detect import detect_objects

K = np.array([[925.46, 0, 702.68], [0, 938.2, 259.91], [0, 0, 1]])
D = np.array([[-0.36083, 0.14103, -0.00837, 0.00406, 0.00562]])
S_rect = np.array([1359, 406])


def undistorted(img: np.ndarray, K, D):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    dst = cv2.undistort(img, K, D, None, newcameramtx)
    x, y, w, h = roi
    assert w == S_rect[0]
    assert h == S_rect[1]
    return dst[y:y + h, x:x + w]


def interactive(seq):
    # parameters for stereo
    stereo = init_stereo(use_sgbm=True)
    _, _, _, _, Q, _, _ = cv2.stereoRectify(K, D, K, D, S_rect, R_rect, np.array([-baseline, 0., 0.]))

    # retrained model
    model = YOLO("runs/detect/train5/weights/best.pt")

    # kalman filters
    kf_manager = KalmanFilterManager()
    initial_pred = True

    for (raw_left, raw_right), (true_rec_left, true_rec_right) in zip(load_stereo_images("raw_data", seq),
                                                                      load_stereo_images("rec_data", seq)):
        # rectify using calibration
        rec_left, rec_right = undistorted(raw_left, K, D), undistorted(raw_right, K, D)

        # send rec_left to YOLO for detection
        measured_roi_list = detect_objects(model, rec_left)

        # update kalman filters
        if initial_pred:
            kf_manager.initialize_filters(measured_roi_list)
            initial_pred = False
        else:
            kf_manager.update(measured_roi_list)

        # get kalman filters prediction
        prediction_roi_list = kf_manager.get_predictions_list()

        # get disparity
        disparity = get_disparity(stereo, rec_left, rec_right)

        # get x, y, z
        estimated_position = []
        for i, (class_id, x, y, w, h) in enumerate(prediction_roi_list):
            estimated_position.append(get_roi_xyz(disparity, (y, y + h, x, x + w), Q))

        # TODO no z if no measurements...

        # TODO display stuff (true and estimated)
        cv2.imshow("raw_left", raw_left)
        cv2.imshow("rec_left", rec_left)
        cv2.imshow("true_rec_left", true_rec_left)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", help="The sequence",
                        choices=["seq_01", "seq_02", "seq_03"], default="seq_01")
    args = parser.parse_args()

    interactive(args.sequence)


if __name__ == "__main__":
    main()
