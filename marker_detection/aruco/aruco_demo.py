import cv2
import cv2.aruco as aruco
import numpy as np

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
aruco_param = aruco.DetectorParameters_create()

aruco_imgs = [
    (cv2.imread("data/aruco_near.jpg", cv2.IMREAD_COLOR), "Aruco Near"),
    (cv2.imread("data/aruco_med.jpg", cv2.IMREAD_COLOR), "Aruco Medium"),
    (cv2.imread("data/aruco_far.jpg", cv2.IMREAD_COLOR), "Aruco Far"),
]


for color_img, label in aruco_imgs:
    greyscale_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
    bounding_boxes, ids, _ = aruco.detectMarkers(greyscale_img, aruco_dict)
    aruco.drawDetectedMarkers(color_img, bounding_boxes)
    cv2.imwrite(f"outputs/{label}.png", color_img)

