# Crane Stabilization



## Installation

Step 1: Follow instructions to install [pypylon](https://github.com/basler/pypylon) to use the Basler Pylon Python wrapper.

Step 2: Follow instructions to install [yumipy](https://github.com/BerkeleyAutomation/yumipy) to use the YuMi Python wrapper developed by AutoLab, UC Berkeley.

Step 3: Install other dependencies by running:

```bash
$ pip install -r requirements.txt
```

## Usage

To run the main algorithm, once the Basler camera and YuMi have been properly connected, simply run:
```bash
$ python main.py
```

Note that as yumipy is only supported for Python 2.7, all files in this repository were developed and tested in Python 2.7, although some testing has been done in Python 3.6. 

There are several settings which can be adjusted in the `main.py` file. 

```python
# Settings
USE_YUMI = True # Toggle between real-time use or experiemnting with YuMi
FILTER_MODE = 3  # 0=2D, 1=3D, 2=Depth, 3=Given
SAVE_VIDEO = True
SAVE_POSE = True # Note ground truth is only availible in YuMi mode
SHOW_MOSAIC = True
SHOW_APRILTAG_DETECTION = True
```
If the `SAVE_VIDEO` and/or `SAVE_POSE` booleans are toggled `True`, the output files will be output into the `data` folder.


There are a couple of other useful scripts that can be used.

To move the YuMi using the specified trajectory independantly from the image stabilization algorithm in `main.py`, run:
```bash
$ python RobotYumi.py
```

To visualise a 3D animation of the CSV waypoints, run:
```bash
$ python visualise_waypoints.py
```

To [calibrate](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html) a camera from images in the folder `data/calibration` of a checkerboard to obtain extrinsic camera parameters and distortion parameters, run:
```bash
$ python calibrate_camera.py
```

