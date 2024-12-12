# PfAS-Project

34759 Perception for Autonomous Systems - Final Project

# Setup

Add the raw images in the `raw_data` folder, and rectified data in the `rec_data` folder.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Content

- `camera_calib.ipynb`, to compute the camera calibration matrices, mainly written
  by [Lkjaersgaard](https://github.com/Lkjaersgaard)
- `load.py`, to properly load the data, mainly written by [Seb-sti1](https://github.com/Seb-sti1)
- `depth.py`, to compute the disparity/depth associated to the stereo images, mainly written
  by [Seb-sti1](https://github.com/Seb-sti1)
- `viz.py`, visualize the data (images, pcd...), mainly written by [Seb-sti1](https://github.com/Seb-sti1)
- `detect.py`, `viz_yolo.py`, compare the result of the YOLO with the truth, mainly written
  by [nandortakacs](https://github.com/nandortakacs)
- `train.py`, train the YOLO model, mainly written by [nandortakacs](https://github.com/nandortakacs)
- `object_detection.py`, try several clustering algorithm to detect relevant objects, mainly written
  by [Seb-sti1](https://github.com/Seb-sti1)
- `kalman_filter_z.py` and `kalman_filter_2d.py`, kalman filter implementation, mainly written 
  by [nandortakacs](https://github.com/nandortakacs)
- `main.py`, use the implemented functions in the other files to test to solve the whole problem, mainly written
  by [Seb-sti1](https://github.com/Seb-sti1)