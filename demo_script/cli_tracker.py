import argparse
import glob
import os.path
import json

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from tracking_util import world_pos_from_image, load_cal_data
from drawing_util import draw_axis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src", type=str, help='The source folder containing the images and calibration data to '
                                                      'use', required=True)
    parser.add_argument("--no-crop", help="Prevents the image from being cropped after distorting", action="store_true")
    parser.add_argument("--show", help="Displays the image before exiting", action="store_true")

    parser.add_argument("--kmatrix", type=str, help="The path to the npy camera matrix")
    parser.add_argument("--dcoeff", type=str, help="The path the the npy distortion coefficient vector")
    parser.add_argument("--file-name", help="The name of the file in the folder to run on",)
    parser.add_argument("--layout", type=str, help='The marker layout json file')

    args = parser.parse_args()

    if args.file_name:
        paths = [os.path.join(args.src, args.file_name)]
    else:
        paths = glob.glob(os.path.join(args.src, "*.jpg"))
        paths += glob.glob(os.path.join(args.src, "*.png"))

    if args.layout:
        layout_path = args.layout
    else:
        layout_path = os.path.join(args.src, 'marker_layout.json')

    if args.kmatrix:
        kmatrix_path = args.kmatrix
    else:
        kmatrix_path = os.path.join(args.src, 'camera_matrix.npy')

    if args.dcoeff:
        dcoeff_path = args.kmatrix
    else:
        dcoeff_path = os.path.join(args.src, 'distortion_coefficients.npy')

    with open(layout_path) as f:
        layout_json = json.load(f)

    # Process the data from the marker layout json file into the format that world_pos_from_image needs
    marker_layout = {int(x["id"]): [
        [x["x"], x["y"], 0],
        [x["x"] + x["size"], x["y"], 0],
        [x["x"] + x["size"], x["y"] + x["size"], 0],
        [x["x"], x["y"] + x["size"], 0]
    ] for x in layout_json}

    marker_pos = {int(x["id"]): {
        "scale": x["size"],
        "pos": np.array([x["x"], x["y"], 0]).reshape(3, 1)
    } for x in layout_json}

    # Run on all images in the src directory
    for path in paths:
        file_name = os.path.basename(path)
        print(f"Loading {file_name}...")

        image = cv.imread(path, cv.IMREAD_COLOR)
        dim = image.shape[:-1][::-1]
        # Inefficiently loading calibration data each time, but it doesn't matter in this case
        # Done since image dimensions *might* be different so we should recalculate the camera matrix just in case
        # and I don't want to write a separate function to do that since it would be of marginal value
        dist_coefficients, new_camera_mtx, roi, mapx, mapy = load_cal_data(kmatrix_path, dcoeff_path, dim)

        # Undistorted the image
        image = cv.remap(image, mapx, mapy, cv.INTER_LINEAR)
        x, y, w, h = roi
        # Crop image to the region of interest
        if not args.no_crop:
            image = image[y:y + h, x:x + w]

        # Aruco detection runs on BW images
        bw_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Grab the translation and rotation vectors plus the ids of the found makers for debugging
        success, rvec, tvec, ids = world_pos_from_image(bw_img, marker_layout, new_camera_mtx, image)

        # Failed to find any markers
        if not success:
            print("failed to find markers")
            if args.show:
                cv.imshow(file_name, image)
                cv.waitKey(0)
                cv.destroyAllWindows()
            continue

        # Use scipy's rotation class to simplify some things
        rot = Rotation.from_rotvec(rvec.reshape(3, ))
        # Grab standard euler angles
        euler = rot.as_euler('zyx', degrees=True)

        print(f"Translation\n"
              f"\t  {np.linalg.norm(tvec):+8.2f} cm\n"
              f"\tX:{tvec[0, 0]:+8.2f} cm\n"
              f"\tY:{tvec[1, 0]:+8.2f} cm\n"
              f"\tZ:{tvec[2, 0]:+8.2f} cm\n"
              f"Rotation\n"
              f"\tZ:{euler[0]:+8.2f} deg\n"
              f"\tY:{euler[1]:+8.2f} deg\n"
              f"\tX:{euler[2]:+8.2f} deg")

        if args.show:
            for marker_id in ids:
                marker_tvec = marker_pos[marker_id]["pos"]
                marker_scale = marker_pos[marker_id]["scale"]
                rot_m = rot.as_matrix()
                draw_axis(image, rvec, tvec + rot_m @ marker_tvec, new_camera_mtx, marker_scale, 2)
            cv.imshow(file_name, image)
            cv.waitKey(0)
            cv.destroyAllWindows()
