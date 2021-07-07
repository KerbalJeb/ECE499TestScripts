import cv2 as cv
import cv2.aruco as aruco
import numpy as np

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)


def load_cal_data(camera_data_path, dist_data_path, dim, alpha=1):
    """
    Loads the camera calibration data from a file
    :param camera_data_path:
    :param dist_data_path:
    :param dim:
    :param alpha:
    :return:
    """
    # https://stackoverflow.com/questions/39432322/what-does-the-getoptimalnewcameramatrix-do-in-opencv
    camera_mtx = np.load(camera_data_path)
    dist_coefficients = np.load(dist_data_path)
    new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(camera_mtx, dist_coefficients, dim, alpha)
    mapx, mapy = cv.initUndistortRectifyMap(camera_mtx, dist_coefficients, None, new_camera_mtx, dim, 5)

    return dist_coefficients, new_camera_mtx, roi, mapx, mapy


def find_markers(image):
    """
    Find the location of the markers on an image
    :param image: A greyscale image
    :return: A tuple containing a list of bounding boxes, a list of marker ids
    """

    # todo: look into implementing ourselves to see if we can get better performance with over exposed images
    boxes, ids, _ = aruco.detectMarkers(image, aruco_dict)
    return boxes, ids


def world_pos_from_image(image, marker_layout, camera_mtx):
    """
    Takes an image with and aruco marker and returns the world location and position of the camera
    :param image: The image to process
    :param marker_layout: The mapping between marker ids and the coordinates of their corners
    :param camera_mtx: The camera matrix
    :return: A tuple containing a boolean to indicate the status of the operation and
    the rotation vector, translation vector and ids if it was successful
    """

    # todo: look into the different methods that solvePnP provides

    # convert image to greyscale and find the markers
    greyscale_img = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    bounding_boxes, ids = find_markers(image)
    ids = ids.reshape(-1)
    if len(bounding_boxes) > 0:
        world_points = [marker_layout[marker_id] for marker_id in ids if marker_id in marker_layout]
        if world_points:
            world_points = np.array(world_points)
            image_points = np.array(bounding_boxes).reshape(-1, 2)
            _, r_vec, t_vec = cv.solvePnP(world_points, image_points, camera_mtx, (0, 0, 0, 0))
            return True, r_vec, t_vec, ids.reshape(-1)

    return False, None, None, None
    pass
