import cv2
import cv2.aruco as aruco
import numpy as np
import timeit

setup = '''
import cv2
import cv2.aruco as aruco

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
aruco_param = aruco.DetectorParameters_create()
color_img = cv2.imread("data/aruco_near.jpg", cv2.IMREAD_COLOR)
'''

code = '''
greyscale_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
bounding_boxes, ids, _ = aruco.detectMarkers(greyscale_img, aruco_dict)
'''

n = 5000
run_time = timeit.timeit(setup=setup, stmt=code, number=n) / n

print(f"{run_time * 1000} ms")
