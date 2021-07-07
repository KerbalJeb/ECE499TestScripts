import cv2 as cv
import cv2.aruco as aruco
import numpy as np

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)


def find_markers(image):
    """
    Find the location of the markers on an image
    :param image: A greyscale image
    :return: A tuple containing a list of bounding boxes, a list of marker ids
    """

    # todo: look into implementing ourselves to see if we can get better performance with over exposed images
    boxes, ids = aruco.detectMarkers(image, aruco_dict)
    return boxes, ids


def world_pos_from_image(image, marker_layout, camera_mtx):
    """
    Takes an image with and aruco marker and returns the world location and position of the camera
    :param image: The image to process
    :param marker_layout: The mapping between marker ids and the coordinates of their corners
    :param camera_mtx: The camera matrix
    :return:
    """

    # todo: track multiple markers using the marker layout parameter to get their world coordinates
    # todo: look into the different methods that solvePnP provides

    # convert image to greyscale and find the markers
    greyscale_img = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    bounding_boxes, ids = find_markers(image)
    if len(bounding_boxes) > 0:
        bounding_box = bounding_boxes[0][0]
        world_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
        _, r_vec, t_vec = cv.solvePnP(world_points, bounding_box, camera_mtx, (0, 0, 0, 0))
        return True, r_vec, t_vec
    else:
        return False, None, None
    pass
