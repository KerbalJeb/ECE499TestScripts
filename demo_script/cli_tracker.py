import argparse
import glob
import os.path
import json

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from tracking_util import world_pos_from_image, load_cal_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src", type=str, help='The image or folder to process', required=True)
    parser.add_argument("-l", "--layout", type=str, help='The marker layout json file', required=True)
    parser.add_argument("-r", help="Runs on all image files in src folder if present", action="store_true")
    parser.add_argument("--no-crop", help="Prevents the image from being cropped after distorting", action="store_true")
    parser.add_argument("--show", help="Displays the image before exiting", action="store_true")
    parser.add_argument("--draw-axis", help="Draws axis on the markers", action="store_true")

    parser.add_argument("--kmatrix", type=str, default="calibration_data/camera_matrix.npy",
                        help="The path to the npy camera matrix")
    parser.add_argument("--dcoeff", type=str, default="calibration_data/distortion_coefficients.npy",
                        help="The path the the npy distortion coefficient vector")
    args = parser.parse_args()

    if args.r:
        paths = glob.glob(os.path.join(args.src, "*.jpg"))
        paths += glob.glob(os.path.join(args.src, "*.png"))
    else:
        paths = [args.src]

    with open(args.layout) as f:
        layout_json = json.load(f)

    marker_layout = {int(x["id"]): [
        [x["x"]            , x["y"]            , 0],
        [x["x"] + x["size"], x["y"]            , 0],
        [x["x"] + x["size"], x["y"] + x["size"], 0],
        [x["x"]            , x["y"] + x["size"], 0]
    ] for x in layout_json}

    for path in paths:
        file_name = os.path.basename(path)
        print(f"Loading {file_name}...")

        image = cv.imread(path, cv.IMREAD_COLOR)
        dim = image.shape[:-1][::-1]
        dist_coefficients, new_camera_mtx, roi, mapx, mapy = load_cal_data(args.kmatrix, args.dcoeff, dim)

        image = cv.remap(image, mapx, mapy, cv.INTER_LINEAR)
        x, y, w, h = roi
        if not args.no_crop:
            image = image[y:y + h, x:x + w]

        bw_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        success, rvec, tvec, ids = world_pos_from_image(image, marker_layout, new_camera_mtx)

        rot = Rotation.from_rotvec(rvec.reshape(3, ))
        euler = rot.as_euler('zyx', degrees=True)

        print(f"Translation: {np.linalg.norm(tvec):+8.2f} cm\n"
              f"\tX:{tvec[0, 0]:+8.2f} cm\n"
              f"\tY:{tvec[1, 0]:+8.2f} cm\n"
              f"\tZ:{tvec[2, 0]:+8.2f} cm\n"
              f"Rotation\n"
              f"\tZ:{euler[0]:+8.2f} deg\n"
              f"\tY:{euler[1]:+8.2f} deg\n"
              f"\tX:{euler[2]:+8.2f} deg")

        if args.show:
            cv.imshow(file_name, image)
            cv.waitKey(0)
            cv.destroyAllWindows()
