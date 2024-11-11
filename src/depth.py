import threading
import time
from typing import Union, Iterator, Tuple

import cv2
import numpy as np
import open3d as o3d
from numpy import ndarray

from load import load_stereo_images

center = None


def __show_pcd__(cloud, should_show):
    global center

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)
    vis.add_geometry(cloud)
    view_control = vis.get_view_control()
    view_control.set_zoom(0.17)
    if center is None:
        center = cloud.get_center() + np.array([-7., 0., 0.])
    view_control.set_lookat(center)

    while not should_show.is_set():  # Keep running until stop_event is set
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)
    vis.destroy_window()


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
    focal_length = (6.900000e+02 + 2.471364e+02) / 2
    Q = np.float32([[1, 0, 0, -w / 2.0],
                    [0, -1, 0, h / 2.0],
                    [0, 0, 0, -focal_length],
                    [0, 0, 1, 0]])

    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    mask_map = disparity > disparity.min()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d[mask_map].astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors[mask_map].astype(np.float64) / 255.0)
    return pcd


def get_stereo_image_disparity_pcd(sequence: str, stereo: Union[cv2.StereoSGBM, cv2.StereoBM]) -> Iterator[
    Tuple[ndarray, ndarray, ndarray, o3d.geometry.PointCloud]]:
    """

    :param sequence:
    :param stereo:
    :return:
    """
    for i, (rec_left, rec_right) in enumerate(load_stereo_images("rec_data", sequence)):
        disparity = get_depth_image(stereo, rec_left, rec_right)
        pcd = depth_to_pcd(rec_left, disparity)

        yield rec_left, rec_right, disparity, pcd


def main(sequence):
    stereo = init_stereo(use_sgbm=True)

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

        should_show = threading.Event()
        pcd_thread = threading.Thread(target=__show_pcd__, args=(cloud, should_show))
        pcd_thread.start()

        k = cv2.waitKey(0)
        should_show.set()
        pcd_thread.join()
        if k & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("seq_01")
