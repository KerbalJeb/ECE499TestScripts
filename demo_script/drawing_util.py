import cv2 as cv
import numpy as np


def draw_axis(img, rotV, t, K):
    """
    Draws a unit axis on the image with the specified rotation and translation
    :param img: The image to draw the axis on
    :param rotV: The rotation vector for the axis (unit vector pointing in the same dir as axis)
    :param t: The translation of the axis relative to the origin
    :param K: The camera matrix
    """
    # https://stackoverflow.com/questions/30207467/how-to-draw-3d-coordinate-axes-with-opencv-for-face-pose-estimation
    points = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    axisPoints = axisPoints.astype(np.int32)
    cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
    cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
    cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)


def draw_movement_widget(terror, img, arrow_scale=500, max_arrow_len=150, widget_center=(200, 200)):
    """
    Draws the movement widget on the image
    :param terror: The translation error as a 3D vector
    :param img: The image to draw on
    :param arrow_scale: Scaling factor for the length of the arrows
    :param max_arrow_len: The maximum arrow length
    :param widget_center: The center of the movement widget as a tuple of pixel coordinates
    :return:
    """
    terror = np.sign(terror) * np.log10(np.abs(terror) + 1)
    error_x = int(np.clip(terror[0] * arrow_scale, -max_arrow_len, max_arrow_len))
    error_y = int(np.clip(terror[1] * arrow_scale, -max_arrow_len, max_arrow_len))
    error_z = int(
        np.clip(terror[2] * arrow_scale / np.sqrt(2), -max_arrow_len / np.sqrt(2), max_arrow_len / np.sqrt(2)))

    widget_x, widget_y = widget_center
    cv.rectangle(img, (widget_x - max_arrow_len, widget_y - max_arrow_len),
                 (widget_x + max_arrow_len, widget_y + max_arrow_len), (255, 255, 255), -1)
    cv.arrowedLine(img, widget_center, (widget_x + error_x, widget_y), (255, 0, 0), 3)
    cv.arrowedLine(img, widget_center, (widget_x, widget_y + error_y), (0, 255, 0), 3)
    cv.arrowedLine(img, widget_center, (widget_x + error_z, widget_y + error_z), (0, 0, 255), 3)
