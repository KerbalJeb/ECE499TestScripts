# ECE499TestScripts

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