from typing import Union, Iterator, Tuple

import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from numpy import ndarray

from load import load_stereo_images, load_calib_matrix, load_labels
from viz import DynamicO3DWindow

baseline = 0.54 * 0.6825199  # 0.68 is a coef to fix the reprojection error
height = 1.4
S = load_calib_matrix("S", (2,)).astype(np.uint32)
K = load_calib_matrix("K", (3, 3))
D = load_calib_matrix("D", (1, 5))
R = load_calib_matrix("R", (3, 3))
T = load_calib_matrix("T", (3, 1))
S_rect = load_calib_matrix("S_rect", (2,)).astype(np.uint32)
P_rect = load_calib_matrix("P_rect", (3, 4))
R_rect = load_calib_matrix("R_rect", (3, 3))


def init_stereo(use_sgbm=True) -> Union[cv2.StereoSGBM, cv2.StereoBM]:
    if use_sgbm:
        window_size = 3
        min_disp = 6
        num_disp = 112 - min_disp
        # TODO tweak params
        return cv2.StereoSGBM_create(minDisparity=min_disp,
                                     numDisparities=num_disp,
                                     blockSize=16,
                                     P1=8 * 3 * window_size ** 2,
                                     P2=32 * 3 * window_size ** 2,
                                     disp12MaxDiff=1,
                                     uniquenessRatio=10,
                                     speckleWindowSize=100,
                                     speckleRange=32
                                     )
    else:
        min_disp = 7
        num_disp = 4 * 16
        block_size = 17
        stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block_size)
        stereo.setMinDisparity(min_disp)
        stereo.setDisp12MaxDiff(200)
        stereo.setUniquenessRatio(11)
        stereo.setSpeckleRange(5)
        stereo.setSpeckleWindowSize(5)
        return stereo


def get_disparity(stereo: Union[cv2.StereoSGBM, cv2.StereoBM],
                  image_left: ndarray, image_right: ndarray, scale: float = 1) -> ndarray:
    """
    Partly from https://github.com/opencv/opencv/blob/6f8c3b13d8c2a4f79c9fc207b416095bb07f317f/samples/python/stereo_match.py#L45

    :param scale:
    :param stereo: the configured stereo algorithm (e.g. using cv2.StereoSGBM_create)
    :param image_left:
    :param image_right:
    :return:
    """
    if scale != 1:
        image_left = cv2.resize(image_left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        image_right = cv2.resize(image_right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)

    disparity_map = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    return disparity_map


def depth_to_pcd(image_left: ndarray, disparity: ndarray, scale: float = 1) -> o3d.geometry.PointCloud:
    # Prepare the color image
    if scale != 1:
        image_left = cv2.resize(image_left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    colors = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)

    # Perspective transformation matrix
    _, _, _, _, Q, _, _ = cv2.stereoRectify(K, D, K, D, S_rect, R, np.array([baseline, 0., 0.]))
    points = cv2.reprojectImageTo3D(disparity, Q)

    crop_mask = np.zeros(disparity.shape, dtype=np.bool_)
    crop_mask[:, :] = False
    crop_mask[150:370, 400:550] = True
    mask_map = disparity > disparity.min()
    mask_map = np.bitwise_and(mask_map, crop_mask)
    points = points[mask_map].astype(np.float64)

    colors = colors[mask_map].astype(np.float64) / 255.0
    # colors[points[:, 2] < -8.2] = [100, 100, 100]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.transform([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd


def get_stereo_image_disparity(sequence: str, stereo: Union[cv2.StereoSGBM, cv2.StereoBM]) -> Iterator[
    Tuple[ndarray, ndarray, ndarray]]:
    """
    :param sequence: the sequence of images
    :param stereo: the stereo algorithm to use
    :return: an iterator on [the rectified left image, rectified right image, disparity]
    """
    for i, (rec_left, rec_right) in enumerate(load_stereo_images("rec_data", sequence)):
        yield rec_left, rec_right, get_disparity(stereo, rec_left, rec_right)


def get_stereo_image_disparity_pcd(sequence: str, stereo: Union[cv2.StereoSGBM, cv2.StereoBM]) -> Iterator[
    Tuple[ndarray, ndarray, ndarray, o3d.geometry.PointCloud]]:
    """
    :param sequence: the sequence of images
    :param stereo: the stereo algorithm to use
    :return: an iterator on [the rectified left image, rectified right image, disparity, point cloud]
    """
    for i, (rec_left, rec_right, disparity) in enumerate(get_stereo_image_disparity(sequence, stereo)):
        yield rec_left, rec_right, disparity, depth_to_pcd(rec_left, disparity)


def get_roi_disparity(disparity: np.ndarray, min_of_disparity: float, roi: Tuple[int, int, int, int]):
    disparity_roi = disparity[roi[0]:roi[1], roi[2]: roi[3]]
    disparity_roi_values = disparity_roi[disparity_roi > min_of_disparity].flatten()

    hist, bin_edges = np.histogram(disparity_roi_values, bins=30)
    peak_index = np.argmax(hist)
    peak_value = (bin_edges[peak_index] + bin_edges[peak_index + 1]) / 2
    # TODO try a (weight) average of the value close to the peak

    # plt.gray()
    # plt.imshow(disparity_roi/disparity.max())
    # plt.show()
    #
    # plt.hist(disparity_roi_values, bins=50, color='blue', edgecolor='black', alpha=0.7)
    # plt.scatter([peak_value], [5000])
    # plt.xlabel("disparity (pixel)")
    # plt.ylabel("count")
    # plt.show()
    return peak_value


def test_with_ground_truth(sequence, show=False):
    # project coordinate in 3D
    stereo = init_stereo(use_sgbm=True)
    _, _, _, _, Q, _, _ = cv2.stereoRectify(K, D, K, D, S_rect, R_rect, np.array([-baseline, 0., 0.]))

    # monitoring arrays
    disparity_vs_z = []
    monitoring = []
    mean_error = []

    # ground truth
    labels_pd = load_labels("rec_data", sequence)
    colors = {"Pedestrian": (255, 0, 0),
              "Cyclist": (0, 255, 0),
              "Car": (0, 0, 255)}
    for i, (rec_left, rec_right, disparity) in enumerate(get_stereo_image_disparity(sequence, stereo)):
        current_labels = labels_pd[labels_pd["frame"] == i]

        d = disparity / disparity.max()
        mean_error_i = np.array([0., 0., 0.])
        for index, row in current_labels.iterrows():
            top_left = (int(row["bbox_left"]), int(row["bbox_top"]))
            bot_right = (int(row["bbox_right"]), int(row["bbox_bottom"]))
            roi_center = (int(top_left[0] / 2 + bot_right[0] / 2), int(top_left[1] / 2 + bot_right[1] / 2))
            x, y, z = row["x"], row["y"], row["z"]

            if z > 10:
                continue

            disparity_roi = get_roi_disparity(disparity, disparity.min(),
                                              (top_left[1], bot_right[1], top_left[0], bot_right[0]))
            xyzw = Q @ [roi_center[0], roi_center[1], disparity_roi, 1]
            xyz = xyzw[:3] / xyzw[3][np.newaxis]
            # xyz[1] += height

            # tracking values to show performance plots
            mean_error_i += abs(xyz - [x, y, z]) / len(current_labels)
            if row["track id"] == 1:
                monitoring.append(
                    [x, y, z,
                     xyz[0], xyz[1], xyz[2],
                     disparity_roi])
            disparity_vs_z.append([disparity_roi, z])

            if show:
                cv2.rectangle(rec_left, top_left, bot_right, colors[row["type"]], 3)

        mean_error.append(mean_error_i)
        if show:
            cv2.imshow("true label left",
                       cv2.resize(rec_left, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR))

            cv2.imshow("disparity",
                       cv2.resize(d, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_LINEAR))
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    disparity_vs_z = np.array(disparity_vs_z)
    min_d, max_d = 1 / 0.028, 1 / 0.014
    outliers = np.argwhere((disparity_vs_z[:, 0] > max_d) | (disparity_vs_z[:, 0] < min_d)).ravel()
    exclude_outliers = np.argwhere((min_d <= disparity_vs_z[:, 0]) & (disparity_vs_z[:, 0] <= max_d)).ravel()
    inv_disparity = 1 / disparity_vs_z[:, 0]

    fig = plt.figure()
    plt.scatter(inv_disparity[exclude_outliers], disparity_vs_z[exclude_outliers, 1], label="Data points", s=3)
    plt.scatter(inv_disparity[outliers], disparity_vs_z[outliers, 1], color="red", label="Outliers", s=3)
    p = np.polyfit(inv_disparity[exclude_outliers], disparity_vs_z[exclude_outliers, 1], 1)
    x = np.linspace(inv_disparity[exclude_outliers].min() * 0.95, inv_disparity[exclude_outliers].max() * 1.05)
    plt.plot(x, x * p[0] + p[1], color="green", label="Fitted line")
    plt.plot(x, x * baseline * K[0, 0], color="yellow", label="Expected line")
    plt.ylabel("z (m)")
    plt.xlabel("Inverse of the disparity (1/pixels)")
    plt.legend()
    plt.grid()
    fig.tight_layout()
    plt.show()
    print(p[0], p[1])
    print(baseline * K[0, 0])

    fig = plt.figure()
    mean_error = np.array(mean_error)
    plt.plot(mean_error, label=["Along x-axis",
                                "Along y-axis",
                                "Along z-axis"])
    plt.xlabel("Image number")
    plt.ylabel("Average distance from estimated to ground truth (m)")
    plt.legend()
    plt.grid()
    plt.ylim([0, mean_error.max() * 1.05])
    plt.xlim([0, mean_error.shape[0] - 1])
    fig.tight_layout()
    plt.show()

    fig = plt.figure()
    monitoring = np.array(monitoring)
    plt.plot(monitoring[:, 3], monitoring[:, 4], label="Estimated trajectory")
    plt.plot(monitoring[:, 0], monitoring[:, 1], label="Ground truth")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.grid()
    fig.tight_layout()
    plt.show()

    fig = plt.figure()
    plt.plot(monitoring[:, 5], label="Estimated trajectory")
    plt.plot(monitoring[:, 2], label="Ground truth")
    plt.xlabel("Image number")
    plt.ylabel("z (m)")
    plt.xlim([0, monitoring.shape[0] - 1])
    plt.legend()
    plt.grid()
    fig.tight_layout()
    plt.show()

    fig = plt.figure()
    plt.plot(monitoring[:, 6])
    plt.grid()
    fig.tight_layout()
    plt.show()


def show_disparity_and_pcd(sequence):
    stereo = init_stereo(use_sgbm=True)
    vis = DynamicO3DWindow()

    for i, (rec_left, rec_right, disparity, pcd) in enumerate(get_stereo_image_disparity_pcd(sequence, stereo)):
        cv2.imshow("left and right",
                   cv2.resize(np.concatenate([rec_left, rec_right], axis=1),
                              None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR))
        cv2.imshow("disparity",
                   cv2.resize(disparity / disparity.max(),
                              None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR))
        cv2.imshow("disparity used",
                   cv2.resize(np.ones_like(disparity) * (disparity > disparity.min()),
                              None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR))

        cloud = pcd.voxel_down_sample(voxel_size=0.05)

        vis.show_pcd(cloud)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    vis.finish()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # show_disparity_and_pcd("seq_01")
    test_with_ground_truth("seq_01")
