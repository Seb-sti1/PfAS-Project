from typing import Union, Iterator, Tuple

import cv2
import numpy as np
import open3d as o3d
from numpy import ndarray

from load import load_stereo_images, load_calib_matrix
from viz import DynamicO3DWindow

baseline = 0.54
S = load_calib_matrix("S", (1, 2))
K = load_calib_matrix("K", (3, 3))
D = load_calib_matrix("D", (1, 5))
R = load_calib_matrix("R", (3, 3))
T = load_calib_matrix("T", (3, 1))
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


def get_depth_image(stereo: Union[cv2.StereoSGBM, cv2.StereoBM],
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
    h, w = image_left.shape[:2]

    """
    0 0 Pedestrian 0 1 0.727451 466.194319 139.161762 557.194320 332.842544 1.744482 0.520582 0.834498 -0.875495 1.374252 6.816363 0.607547
    0 1 Pedestrian 0 0 0.612450 389.158096 150.885617 497.158096 359.917155 1.625074 0.630655 0.721248 -1.333895 1.397117 5.923950 0.404248
    0 5 Pedestrian 0 1 1.321152 485.398585 145.273299 502.065252 200.312156 1.677151 0.446023 0.749989 -3.436733 0.602339 21.976265 1.170425
    0 6 Pedestrian 0 1 1.662855 525.723004 137.996529 541.389670 191.857804 1.952553 0.628458 0.744739 -2.640812 0.409017 26.109648 1.565453
    0 7 Pedestrian 0 0 1.589458 545.146393 143.123321 559.813059 190.469783 1.709764 0.597256 0.524796 -1.905936 0.357788 25.902194 1.519108
    0 8 Pedestrian 0 0 1.443567 568.546716 145.927759 581.213382 191.540706 1.590542 0.464626 0.670820 -1.082702 0.383012 25.062125 1.403141
    0 9 Cyclist 0 2 2.067586 505.904313 138.198030 535.341152 183.041223 1.841708 0.508194 1.607527 -3.570178 0.096495 29.949718 1.952429
    """

    # Perspective transformation matrix
    _, _, _, _, Q, _, _ = cv2.stereoRectify(K, D, K, D, S[0, :].astype(np.uint32), R, np.array([baseline, 0., 0.]))
    points = cv2.reprojectImageTo3D(disparity, Q)

    crop_mask = np.zeros(disparity.shape, dtype=np.bool_)
    crop_mask[:, :] = False
    crop_mask[150:370, 400:550] = True
    mask_map = disparity > disparity.min()
    # mask_map = np.bitwise_and(mask_map, crop_mask)
    points = points[mask_map].astype(np.float64)

    # colors = np.zeros_like(points)
    # colors[points[:, 2] < -6.812] = [1., 0., 1.]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors[mask_map].astype(np.float64) / 255.0)
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
        disparity = get_depth_image(stereo, rec_left, rec_right)

        yield rec_left, rec_right, disparity


def get_stereo_image_disparity_pcd(sequence: str, stereo: Union[cv2.StereoSGBM, cv2.StereoBM]) -> Iterator[
    Tuple[ndarray, ndarray, ndarray, o3d.geometry.PointCloud]]:
    """
    :param sequence: the sequence of images
    :param stereo: the stereo algorithm to use
    :return: an iterator on [the rectified left image, rectified right image, disparity, point cloud]
    """
    for i, (rec_left, rec_right, disparity) in enumerate(get_stereo_image_disparity(sequence, stereo)):
        pcd = depth_to_pcd(rec_left, disparity)

        yield rec_left, rec_right, disparity, pcd


def main(sequence):
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
    main("seq_01")
