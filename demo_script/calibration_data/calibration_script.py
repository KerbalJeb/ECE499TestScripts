import cv2
import numpy as np
import glob
from pprint import pprint
import argparse
import os.path

parser = argparse.ArgumentParser()
parser.add_argument("--show", help="Displays the image before exiting", action="store_true")
parser.add_argument("-s", "--src", help="The folder to load images from", required=True, type=str)
parser.add_argument("--dst", help="The folder to load images from", default="./")

args = parser.parse_args()

CHECKERBOARD = (7, 9)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

image_names = glob.glob(os.path.join(args.src, "*.jpg"))
image_names += glob.glob(os.path.join(args.src, "*.png"))

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
gray = None
for fname in image_names:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)
    # If found, add object points, image points (after refining them)
    if ret is True:
        objpoints.append(objp)
        # Iteratively improves the accuracy of the corner detection
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
        if args.show:
            cv2.imshow('img', img)
            cv2.waitKey(0)

if gray is not None:
    # Calculate Parameters
    result, camera_matrix, distortion_coefficients, _, _ = \
        cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    h, w = gray.shape[:2]

    np.save(os.path.join(args.dst, 'camera_matrix.npy'), camera_matrix)
    np.save(os.path.join(args.dst, 'distortion_coefficients.npy'), distortion_coefficients)

    print("Camera Matrix:")
    pprint(camera_matrix)
    print("Distortion Coefficients:")
    pprint(distortion_coefficients)
cv2.destroyAllWindows()
