# ECE499 Smart Marine Charger Sensor Project

This project consists of three main Python scripts: a camera calibration script, a command line tool and a GUI tool. The
GUI tool will process live data from a web camera and the command line tool can be run on a folder of images for testing
purposes.

## Getting Started

In order to use this project you will need to

1. Install Python + dependencies
1. Perform camera calibration
1. Generate marker layout JSON files

These steps are described in more detail below

### Installation

Install Python 3.8+ by following the direction on: https://www.python.org/. Also ensure that Python and pip have been
installed correctly by typing `python --version` and `pip --version` into a terminal and ensuring that the commands run
correctly. Note: depending on the system you may need to run `python3 --version` if `python --version` gives a 2.x
version of Python.

Next you will need to install the required python libraries, this can be done by
running ` pip install -r requirements.txt ` from the top level directory. The libraries can also be installed in a
virtual environment if you don't want to install them system wide: https://docs.python.org/3/library/venv.html

### Camera Calibration

The camera calibration script can be found in calibration_data/calibration_script.py. Take several images of
a chess board pattern from different angles using the camera to be calibrated and store them as png or jpg files in a
single folder. Then run the script to generate the camera matrix and distortion coefficient files.

Chess board pattern can be found at: https://www.mrpt.org/downloads/camera-calibration-checker-board_9x7.pdf

```text
usage: calibration_script.py [-h] [--show] -s SRC [--dst DST]

optional arguments:
  -h, --help         show this help message and exit
  --show             displays the image before exiting
  -s SRC, --src SRC  the folder to load images from
  --dst DST          the folder to save the calibration data to
```

### Marker Layout Files

The marker layout is described using a JSON file similar to the one shown below. It consists of a list of JSON objects
that each have id, size, x and y properties. Note that every maker present must have a unique ID.

```JSmin
[
// List of all markers present in image
  {
    "id": 0, // marker id
    "size": 9.5, // length of marker edges (cm)
    "x": 0,  // x-offset of marker upper left corrner from origin 
    "y": 0   // y-offset of marker upper left corrner from origin 
  },
  {
    "id": 1,
    "size": 9.5, 
    "x": 10, 
    "y": 10  
  }
]
```

## Command Line Usage

The command line version of this script can be found at cli_tracker.py. It takes one required command line
argument --src, which is a path to a source folder containing the following:

- The images to process (jpg or png)
- The camera matrix and distortion coefficients npy files (named camera_matrix.npy and distortion_coefficients.npy)
  respectively
- The marker layout json file named marker_layout.json

Other command line arguments

```
usage: cli_tracker.py [-h] -s SRC [--no-crop] [--show] [--kmatrix KMATRIX] [--dcoeff DCOEFF] [--file-name FILE_NAME] [--layout LAYOUT]

optional arguments:
  -h, --help            show this help message and exit
  -s SRC, --src SRC     The source folder containing the images and calibration data to use
  --no-crop             Prevents the image from being cropped after distorting
  --show                Displays the image before exiting
  --kmatrix KMATRIX     The path to the npy camera matrix
  --dcoeff DCOEFF       The path the the npy distortion coefficient vector
  --file-name FILE_NAME
                        The name of the file in the folder to run on
  --layout LAYOUT       The marker layout json file

```

## GUI Usage

TODO: Fill out
