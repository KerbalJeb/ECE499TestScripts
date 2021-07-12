# ECE499 Smart Marine Charger Sensor Project
## Camera Calibration

The camera calibration script can be found in demo_script/calibration_data/calibration_script.py. Take several images of
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

## Command Line Usage

The command line version of this script can be found at demo_script/cli_tracker.py. It takes one required command line
argument --src, which is a path to a source folder containing the following:

- The images to process (jpg or png)
- The camera matrix and distortion coefficients as npy files (named camera_matrix.npy and distortion_coefficients.npy)
  respectively
- The marker layout json file named marker_layout.json, format shown below

```JSmin
[
// List of all markers present in image
  {
    "id": 0, // marker id
    "size": 9.5, // length of marker edges (cm)
    "x": 0,  // x-offset of marker upper left corrner from origin 
    "y": 0   // y-offset of marker upper left corrner from origin 
  }
]
```

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
