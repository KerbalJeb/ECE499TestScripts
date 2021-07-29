import cv2 as cv
import numpy as np


def draw_axis(img, rvec, tvec, camera_matrix, scale=1, line_width=3):
    """
    Draws a unit axis on the image with the specified rotation and translation
    :param img: The image to draw the axis on
    :param rvec: The rotation vector for the axis (unit vector pointing in the same dir as axis)
    :param tvec: The translation of the axis relative to the origin
    :param camera_matrix: The camera matrix
    :param scale: The size of the axis
    :param line_width: The width of the axis lines in pixels
    """
    # https://stackoverflow.com/questions/30207467/how-to-draw-3d-coordinate-axes-with-opencv-for-face-pose-estimation
    points = np.float32([[scale, 0, 0], [0, scale, 0], [0, 0, scale], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv.projectPoints(points, rvec, tvec, camera_matrix, (0, 0, 0, 0))
    axisPoints = axisPoints.astype(np.int32)
    # x-axis (red)
    cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (0, 0, 255), line_width)
    # y-axis (green)
    cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), line_width)
    # z-axis (blue)
    cv.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (255, 0, 0), line_width)


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
