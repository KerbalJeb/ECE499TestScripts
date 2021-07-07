import argparse

import cv2 as cv
import numpy as np
import PySimpleGUI as sg


def main():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=int, default=0, help="The id of the camera to use")
    args = parser.parse_args()

    # todo add argument to load image and maybe video files (or it could be a GUI element)
    # todo add batch process mode to load all images from one folder and write the processed images to an output folder

    # Load the camera cal data
    camera_mtx = np.load("calibration_data/camera_matrix.npy")
    dist_coefficients = np.load("calibration_data/distortion_coefficients.npy")

    # Image size
    w = 1920
    h = 1080
    dim = (w, h)

    # How much of the image mapping to keep
    alpha = 1

    # Precompute mappings
    # https://stackoverflow.com/questions/39432322/what-does-the-getoptimalnewcameramatrix-do-in-opencv
    new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(camera_mtx, dist_coefficients, (w, h), alpha)
    mapx, mapy = cv.initUndistortRectifyMap(camera_mtx, dist_coefficients, None, new_camera_mtx, dim, 5)

    # Setup the GUI
    sg.theme('Black')

    # todo add GUI elements to show position and rotation
    # todo add some check boxes to enable/disable some overlays
    # todo add text input for target position and required accuracy

    # define the window layout
    layout = [[sg.Text('OpenCV Demo', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Image(filename='', key='image')]]

    # create the window and show it without the plot
    window = sg.Window('Demo Application - OpenCV Integration',
                       layout, location=(800, 400))

    # Get the camera
    camera_id = args.camera
    cam = cv.VideoCapture(camera_id, cv.CAP_DSHOW)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, w)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, h)

    # Event loop
    while True:
        # todo call functions to find camera pos from markers
        # todo update GUI as needed
        event, values = window.read(timeout=20)
        if event == sg.WINDOW_CLOSED:
            break
        status, frame = cam.read()
        if status:
            # Undistort the image and display it
            frame = cv.remap(frame, mapx, mapy, cv.INTER_LINEAR)
            imgbytes = cv.imencode('.png', frame)[1].tobytes()
            window['image'].update(data=imgbytes)

    cam.release()


if __name__ == "__main__":
    main()
