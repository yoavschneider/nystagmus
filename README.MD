# Bachelor Thesis - Eye Tracking for People With Nystagmus

## Prerequisites
* Python 3.6+
* OpenCV and OpenCV-Contrib [([Version  3.4.1](https://opencv.org/))
* [Dlib](http://dlib.net/)
* [Numpy](http://www.numpy.org/)
* [Pyplot](https://matplotlib.org/api/pyplot_api.html)
* [Pygame](https://www.pygame.org/)
* [PyQT5](https://www.riverbankcomputing.com/software/pyqt/intro) (Python 3)
* [PyQT4](https://www.riverbankcomputing.com/software/pyqt/intro) (Python 2)

* [Tobii Pro Python SDK](http://developer.tobiipro.com/python/python-getting-started.html)

## Installation
* make 
* python3 setup.py build_ext --inplace; python3 main.py -i

## Code
### Iris and Face Detection
* **gradient.pyx** - calculation and thresholding image gradients
* **iris.pyx** - finding pupil center ([Fabian Timm and Erhardt Barth](http://www.inb.uni-luebeck.de/fileadmin/files/PUBPDFS/TiBa11b.pdf))
* **face.pyx** - 2D tracker, finds face, pupils and calculate different phases

### Calibration
* **calibrate_camera.py** - camera intrinsic calibration
* **extrinsic_calibration.py** - camera extrinsic calibration ([K. Takahashi, S. Nobuhara and T. Matsuyama](https://computer-vision.github.io/takahashi2012cvpr/))
* **nyst_calibration.py**, **gaze_loop.pyx** - simple calibration of own implementation for the experimental part
* **tobii_calibration.py**, **gaze_loop.pyx** - simple calibration of the Tobii eye tracker

### 3D Model
* **projection.pyx** - uses calibration data to calculate 3D positions + visualize points in the screen coordinate system
* **visualize.py** - visualizes the calibration results, screen and camera position and orientation

### Experiment
* **tobii_recorder.py** - records data from the Tobii Pro Eye Tracker
* **gaze_recorder.py**, **gaze_recorder.pyx** - records data from own implementations
* **rigid_transform.py** - calculates a rigid transformation ([source](http://nghiaho.com/?page_id=671))
* **analyser.py** - calculates statistics and produce plots
* **plot_nyst_calibration.py** - plots own implementation calibration results

### Utility
* **geo.pyx** - convience geometrical calculations
* **stream.pyx** - a faster asynchronous webcam streamer
