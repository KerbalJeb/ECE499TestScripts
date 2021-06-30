import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation
from pprint import pprint

w = 1920
h = 1080

TARGET_POS = np.array([-0.4, -0.7, 2.5]).reshape(3, 1)
MARGIN = 0.2
WIDGET_X = w - 200
WIDGET_Y = h - 200
WIDGET_CENTER = (WIDGET_X, WIDGET_Y)
ARROW_SCALE_FACTOR = 500
MAX_ARROW_LEN = 150

DISTORTION_COEFFICIENTS = np.array([-0.33265127, 0.10013361, -0.00089593, 0.00123881, -0.00751234])
CAMERA_MATRIX = np.array([
    [1.37760479e+03, 0.00000000e+00, 9.63048231e+02],
    [0.00000000e+00, 1.36239697e+03, 5.59607244e+02],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
])
CAMERA_MATRIX_INV = np.linalg.inv(CAMERA_MATRIX)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
aruco_param = aruco.DetectorParameters_create()

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

i = 0


def draw_axis(img, R, t, K):
    rotV = R
    points = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, -1], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    axisPoints = axisPoints.astype(np.int32)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)
    return img


def draw_movement_widget(e, img):
    e = np.sign(e)*np.log10(np.abs(e) + 1)
    error_x = int(np.clip(e[0] * ARROW_SCALE_FACTOR, -MAX_ARROW_LEN, MAX_ARROW_LEN))
    error_y = int(np.clip(e[1] * ARROW_SCALE_FACTOR, -MAX_ARROW_LEN, MAX_ARROW_LEN))
    error_z = int(
        np.clip(e[2] * ARROW_SCALE_FACTOR / np.sqrt(2), -MAX_ARROW_LEN / np.sqrt(2), MAX_ARROW_LEN / np.sqrt(2)))

    cv2.rectangle(img, (WIDGET_X - MAX_ARROW_LEN, WIDGET_Y - MAX_ARROW_LEN),
                  (WIDGET_X + MAX_ARROW_LEN, WIDGET_Y + MAX_ARROW_LEN), (255, 255, 255), -1)
    cv2.arrowedLine(img, WIDGET_CENTER, (WIDGET_X - error_x, WIDGET_Y), (255, 0, 0), 3)
    cv2.arrowedLine(img, WIDGET_CENTER, (WIDGET_X, WIDGET_Y - error_y), (0, 255, 0), 3)
    cv2.arrowedLine(img, WIDGET_CENTER, (WIDGET_X + error_z, WIDGET_Y + error_z), (0, 0, 255), 3)


while cv2.waitKey(1) != ord('q'):
    ret, frame = cam.read()
    if not ret:
        break

    undistorted_img = cv2.undistort(frame, CAMERA_MATRIX, DISTORTION_COEFFICIENTS, None)
    greyscale_img = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2GRAY)

    bounding_boxes, ids, _ = aruco.detectMarkers(greyscale_img, aruco_dict)

    if len(bounding_boxes) > 0:
        bounding_boxes = bounding_boxes[0][0]
        world_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
        _, rot, trans = cv2.solvePnP(world_points, bounding_boxes, CAMERA_MATRIX, (0, 0, 0, 0))
        draw_axis(undistorted_img, rot, trans, CAMERA_MATRIX)

        error = TARGET_POS - trans
        print(np.linalg.norm(error))

        draw_movement_widget(error, undistorted_img)

        if np.linalg.norm(error) < MARGIN:
            pts = bounding_boxes.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(undistorted_img, [pts], True, (0, 255, 0), 8)

    cv2.imshow('frame', cv2.resize(undistorted_img, None, fx=0.5, fy=0.5))

cam.release()
cv2.destroyAllWindows()