import argparse
from PySimpleGUI.PySimpleGUI import Window

import cv2 as cv
import numpy as np
import PySimpleGUI as sg
import os.path

from tracking_util import load_cal_data


def main():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=int, default=0, help="The id of the camera to use")
    args = parser.parse_args()

    # todo add argument to load image and maybe video files (or it could be a GUI element)
    # todo add batch process mode to load all images from one folder and write the processed images to an output folder

    # Image size
    w = 1920
    h = 1080
    dim = (w, h)

    # How much of the image mapping to keep
    alpha = 1

    # Precompute mappings
    cal_data = load_cal_data("calibration_data/camera_matrix.npy",
                             "calibration_data/distortion_coefficients.npy",
                             dim, alpha)

    dist_coefficients, new_camera_mtx, roi, mapx, mapy = cal_data

    # Setup the GUI
    sg.theme('Black')

    # todo add GUI elements to show position and rotation
    # todo add some check boxes to enable/disable some overlays
    # todo add text input for target position and required accuracy

    # checkbox GUI 
    checks = [[sg.FolderBrowse(key="-FILE-")],
              [sg.Checkbox('Positional data:', default=False, key="-POSITION-")],
              [sg.Button('Submit')]]

    # boolean checks 
    pos = False

    # create checkbox window
    check_window = sg.Window('Select options', checks, size=(300,200))

    # option loop
    while True:
        event, values = check_window.read()
        if event == sg.WIN_CLOSED or event=="Submit":
            break

    if values["-POSITION-"] == True:
        pos = True  

    check_window.close()

    # define the window layout
    layout = [[sg.Text('OpenCV Demo', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Text('', size=(80, 1), justification ='center', font='Helvetica 20', key="-POS-")],
              [sg.Image(filename='', key='image')],]

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
        
        # update positional data 
        if pos:
            window["-POS-"].update('position')
        
        status, frame = cam.read()
        if status:
            # Undistort the image and display it
            frame = cv.remap(frame, mapx, mapy, cv.INTER_LINEAR)
            # todo crop image to only show roi
            #  (https://stackoverflow.com/questions/39432322/what-does-the-getoptimalnewcameramatrix-do-in-opencv)
            imgbytes = cv.imencode('.png', frame)[1].tobytes()
            window['image'].update(data=imgbytes)

    cam.release()


if __name__ == "__main__":
    main()
