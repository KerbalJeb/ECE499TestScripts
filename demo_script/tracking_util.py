import cv2 as cv
import cv2.aruco as aruco
import numpy as np

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)


def load_cal_data(camera_data_path, dist_data_path, dim, alpha=1):
    """
    Loads the camera calibration data from a file
    :param camera_data_path: The path to the camera calibration matrix file
    :param dist_data_path: The path the the distortion coefficient file
    :param dim: The dimensions of the output image
    :param alpha: A value between 0 and 1 that describes how much of the unusable image will be kept
    (1=keep whole image, 0=crop out all invalid areas)
    :return: dist_coefficients, camera matrix, roi, mapx, mapy
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


def world_pos_from_image(image, marker_layout, camera_mtx, draw_img=None):
    """
    Takes an image with aruco markers and returns the world location and position of the camera
    :param draw_img:
    :param image: The image to process
    :param marker_layout: The mapping between marker ids and the coordinates of their corners
    :param camera_mtx: The camera matrix
    :return: A tuple containing a boolean to indicate the status of the operation and
    the rotation vector, translation vector and ids if it was successful
    """

    # todo: look into the different methods that solvePnP provides

    bounding_boxes, ids = find_markers(image)
    # Make the ids a row vector
    if ids is not None:
        found_ids = ids.reshape(-1)
        # Intersection of the found ids and the ids we know about from the marker layout
        ids = np.array([i for i in found_ids if i in marker_layout])
        # Discard bounding boxes that match unknown ids and make sure they still are in the same order as the ids
        bounding_boxes = [bounding_boxes[np.where(found_ids == i)[0][0]] for i in ids]
        # Optionally draw the markers on an image
        if draw_img is not None:
            aruco.drawDetectedMarkers(draw_img, bounding_boxes, ids)
        # Order the marker world positions in the same order as the bounding boxes returned by find_markers (ignoring
        # any unknown ids)
        world_points = [marker_layout[marker_id] for marker_id in ids if marker_id in marker_layout]
        if world_points:
            # Flatten the arrays of bounding boxes to just be lists of points in R^2 and R^3
            world_points = np.array(world_points, dtype=np.float32).reshape(-1, 3)
            image_points = np.array(bounding_boxes, dtype=np.float32).reshape(-1, 2)
            # Run opencv's pose estimation algorithm (iterative by default)
            status, r_vec, t_vec = cv.solvePnP(world_points, image_points, camera_mtx, (0, 0, 0, 0), cv.SOLVEPNP_IPPE)
            # Should not fail to compute transform
            assert status
            return True, r_vec, t_vec, ids

    return False, None, None, None
