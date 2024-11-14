import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from depth import init_stereo, get_stereo_image_disparity_pcd
from viz import DynamicO3DWindow


def find_best_k_silhouette(X, k_list):
    best_km = None
    best_score = None
    for k in tqdm(k_list):
        km = KMeans(n_clusters=k, init='random',
                    n_init=10, max_iter=300, tol=1e-04, random_state=0)
        km.fit(X)
        score = silhouette_score(X, km.labels_)

        if best_km is None or score > best_score:
            best_km, best_score = km, score
    return best_km


def main_bis(sequence):
    cmap = plt.get_cmap("tab20")
    stereo = init_stereo(use_sgbm=True)

    km = KMeans(n_clusters=10, init='random',
                n_init=10, max_iter=300, tol=1e-04, random_state=0)

    for i, (rec_left, _, disparity, pcd) in enumerate(get_stereo_image_disparity_pcd(sequence, stereo)):
        rows, cols = disparity.shape
        min_disparity = disparity.min()
        x, y = np.meshgrid(np.arange(rows), np.arange(cols))
        data = np.column_stack((1 / rows * x.ravel(), 1 / cols * y.ravel(), 5 * disparity.ravel()))
        data[data[:, 2] == min_disparity] = (0, 0, 0)
        labels = km.fit_predict(data)

        # color by label
        max_label = labels.max()
        colors = cmap(labels / (max_label if max_label > 0 else 1))

        cv2.imshow("left", cv2.resize(rec_left,
                                      None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR))
        cv2.imshow("disparity", cv2.resize(disparity / disparity.max(),
                                           None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR))
        cv2.imshow("classification", cv2.resize(colors.reshape(rows, cols, 4),
                                                None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR))
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def main(sequence):
    # TODO try using get_stereo_image_disparity_pcd(sequence, stereo)
    #      to detect blob of points using coordinates
    #      - look at day11 day
    # TODO find attributes on the blob of points
    # TODO use attributes to try to classify blob in relevant category
    # TODO ask is it in real time or is it offline

    cmap = plt.get_cmap("tab20")
    stereo = init_stereo(use_sgbm=True)
    vis = DynamicO3DWindow()

    k_list = range(2, 60, 7)
    for i, (rec_left, _, disparity, pcd) in enumerate(get_stereo_image_disparity_pcd(sequence, stereo)):
        print(f"Size {len(pcd.points)}")
        pcd = pcd.voxel_down_sample(voxel_size=0.1)
        print(f"Size {len(pcd.points)}")

        km = find_best_k_silhouette(pcd.points, k_list)
        labels = km.labels_

        # color by label
        max_label = labels.max()
        colors = cmap(labels / (max_label if max_label > 0 else 1))
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        # show
        cv2.imshow("left", cv2.resize(rec_left,
                                      None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR))
        cv2.imshow("disparity", cv2.resize(disparity / disparity.max(),
                                           None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR))
        vis.show_pcd(pcd)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    vis.finish()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("seq_01")
    # main_bis("seq_01")
