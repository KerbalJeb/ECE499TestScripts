import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import glob
import pytest
import os

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
aruco_param = aruco.DetectorParameters_create()
aruco_param.adaptiveThreshWinSizeMax = 100
aruco_param.perspectiveRemovePixelPerCell = 15
aruco_param.perspectiveRemoveIgnoredMarginPerCell = 0.2


def test_for_markers(valid_ids, path):
    color_img = cv.imread(path, cv.IMREAD_COLOR)
    greyscale_img = cv.cvtColor(color_img, cv.COLOR_RGB2GRAY)
    bounding_boxes, ids, rejected = aruco.detectMarkers(greyscale_img, aruco_dict, parameters=aruco_param)
    aruco.drawDetectedMarkers(color_img, bounding_boxes)
    if ids is None:
        ids = []
    else:
        ids = list(ids.reshape(-1))
    if ids != valid_ids:
        aruco.drawDetectedMarkers(color_img, rejected)
        dir_name = os.path.dirname(path)
        file_name = os.path.basename(path)
        results_path = os.path.join(dir_name, "results")
        if not os.path.isdir(os.path.join(dir_name, results_path)):
            os.mkdir(results_path)
        cv.imwrite(os.path.join(results_path, file_name), color_img)
        assert ids == valid_ids
