import argparse
from typing import Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from depth import init_stereo, get_disparity, baseline, R_rect, get_roi_xyz
from detect import detect_objects
from kalman_filter_z import KalmanFilterManager
from load import load_stereo_images

K = np.array([[925.46, 0, 702.68], [0, 938.2, 259.91], [0, 0, 1]])
D = np.array([[-0.36083, 0.14103, -0.00837, 0.00406, 0.00562]])
S_rect = np.array([1359, 406])


def draw_circle(img, class_id, x_p, y_p, x, y, z, colors, circle_radius=5):
    cv2.circle(img, (x_p, y_p), circle_radius, colors[class_id])
    cv2.putText(img, f"{x:.1f} {y:.2f} {z:.2f}",
                (x_p + circle_radius + 1, y_p),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))


def draw_rectangle(img, class_id, x_p, y_p, w, h, colors):
    cv2.rectangle(img, (x_p - w // 2, y_p - h // 2), (x_p + w // 2, y_p + h // 2), colors[class_id], 2)


def undistorted(img: np.ndarray, K, D):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    dst = cv2.undistort(img, K, D, None, newcameramtx)
    x, y, w, h = roi
    assert w == S_rect[0]
    assert h == S_rect[1]
    return dst[y:y + h, x:x + w]


def xyz_from_roi_center_and_z(center: Tuple[int, int], z: float, f: float) -> Tuple[float, float, float]:
    x_p, y_p = center
    return x_p * z / f, y_p * z / f, z


def interactive(seq):
    # parameters for stereo
    stereo = init_stereo(use_sgbm=True)
    _, _, _, _, Q, _, _ = cv2.stereoRectify(K, D, K, D, S_rect, R_rect, np.array([-baseline, 0., 0.]))

    # retrained model
    model = YOLO("runs/detect/train5/weights/best.pt")

    # kalman filters
    kf_manager = KalmanFilterManager()
    initial_pred = True

    # display params
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for (raw_left, raw_right), (true_rec_left, true_rec_right) in zip(load_stereo_images("raw_data", seq),
                                                                      load_stereo_images("rec_data", seq)):
        # rectify using calibration
        rec_left, rec_right = undistorted(raw_left, K, D), undistorted(raw_right, K, D)
        # rec_left, rec_right = true_rec_left, true_rec_right

        # send rec_left to YOLO for detection
        raw_measured_roi_list = detect_objects(model, rec_left)

        # get disparity
        disparity = get_disparity(stereo, rec_left, rec_right)

        # get x, y, z
        estimated_position = []
        for i, (class_id, x, y, w, h) in enumerate(raw_measured_roi_list):
            estimated_position.append(get_roi_xyz(disparity, (y - h // 2, y + h // 2, x - w // 2, x + w // 2), Q))

        # update kalman filters
        measurements_list = [(class_id, x, y, z)
                             for (class_id, x, y, _, _), (_, _, z)
                             in zip(raw_measured_roi_list, estimated_position)]
        if initial_pred:
            kf_manager.initialize_filters(measurements_list)
            initial_pred = False
        else:
            kf_manager.update(measurements_list)

        # get kalman filters prediction
        prediction_list = [(class_id, x_p, y_p, *xyz_from_roi_center_and_z((x_p, y_p), z, float(K[0, 0])))
                           for (class_id, x_p, y_p, z, _, _) in kf_manager.get_predictions_list()]

        # display (true and estimated)
        cv2.imshow("raw_left", raw_left)

        yolo_rec_left = rec_left.copy()
        for pred in raw_measured_roi_list:
            draw_rectangle(yolo_rec_left, *pred, colors)
        cv2.imshow("yolo detection on rec_left", yolo_rec_left)

        cv2.imshow("disparity", disparity / disparity.max())

        for pred in prediction_list:
            draw_circle(rec_left, *pred, colors)
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
