from typing import Union, Iterator, Tuple

import cv2
import numpy as np
import open3d as o3d
from numpy import ndarray

from load import load_stereo_images, load_calib_matrix
from viz import DynamicO3DWindow

baseline = 0.54
K = load_calib_matrix("K", (3, 3))
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

    # Perspective transformation matrix
    # TODO make sure this is the correct value
    Q = np.float32([
        [1, 0, 0, -K[0, 2]],
        [0, -1, 0, K[1, 2]],
        [0, 0, 0, -K[0, 0]],
        [0, 0, 1 / baseline, 0]
    ])
    points = cv2.reprojectImageTo3D(disparity, Q)
    mask_map = disparity > disparity.min()
    # TODO maybe missing a rotation: the floor is not coplanar (already tried with R_rect)
    points = points[mask_map].astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors[mask_map].astype(np.float64) / 255.0)
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
