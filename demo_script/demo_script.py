import argparse
from PySimpleGUI.PySimpleGUI import Window

import cv2 as cv
import numpy as np
import PySimpleGUI as sg
import os.path

from scipy.spatial.transform import Rotation

from drawing_util import draw_axis
from tracking_util import load_cal_data, world_pos_from_image, load_layout
from gui_helper import event_function

EventFunctions = {}

W = 1920
H = 1080
dim = (W, H)


def main():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=int, default=0, help="The id of the camera to use")
    args = parser.parse_args()

    # todo add argument to load image and maybe video files (or it could be a GUI element)
    # todo add batch process mode to load all images from one folder and write the processed images to an output folder

    # Image size

    loaded_cal = False
    loaded_layout = False

    # Setup the GUI
    sg.theme('Black')

    # todo add GUI elements to show position and rotation
    # todo add some check boxes to enable/disable some overlays
    # todo add text input for target position and required accuracy

    # define the window layout

    control_panel = [
        [sg.Checkbox('Draw Axis', key='draw-axis', default=False)],
        [sg.Checkbox('Draw Marker', key='draw-marker', default=False)],
    ]

    layout = [
        [
            sg.Column([[sg.Text('Calibration Data')],
                       [sg.Input(key='cal-path', enable_events=True),
                        sg.FileBrowse('Browse', file_types=(('npz', '*.npz'),))]]),
            sg.Column([[sg.Text('Marker Layout')],
                       [sg.Input(key='layout-path', enable_events=True),
                        sg.FileBrowse('Browse', file_types=(('json', '*.json'),))]])
        ],
        [sg.Frame('Control Panel', layout=control_panel, vertical_alignment='top'),
         sg.Image(filename='', key='image'),
         sg.Column([
             [sg.Frame('Position', vertical_alignment='top', layout=[
                 [sg.Text('N/A', key='pos', size=(15, 1))],
                 [sg.Text('X: N/A', key='x-pos', size=(15, 1))],
                 [sg.Text('Y: N/A', key='y-pos', size=(15, 1))],
                 [sg.Text('Z: N/A', key='z-pos', size=(15, 1))],
             ])],
             [sg.Frame('Rotation', vertical_alignment='top', layout=[
                 [sg.Text('X: N/A', key='x-rot', size=(15, 1))],
                 [sg.Text('Y: N/A', key='y-rot', size=(15, 1))],
                 [sg.Text('Z: N/A', key='z-rot', size=(15, 1))],
             ])]], vertical_alignment='top')
         ]
    ]

    # create the window and show it without the plot
    window = sg.Window('Demo Application - OpenCV Integration', layout)

    # Get the camera
    camera_id = args.camera
    cam = cv.VideoCapture(camera_id, cv.CAP_DSHOW)

    if not cam.isOpened():
        print("Failed to load camera")
        return

    cam.set(cv.CAP_PROP_FRAME_WIDTH, W)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, H)

    # Event loop
    while True:
        # todo call functions to find camera pos from markers
        # todo update GUI as needed
        event, values = window.read(timeout=20)

        if event == 'cal-path':
            cal_path = values[event]
            if os.path.isfile(cal_path):
                loaded_cal = True
                cal_data = load_cal_data(cal_path, dim, 1)
                dist_coefficients, camera_mtx, roi, mapx, mapy = cal_data

        if event == 'layout-path':
            layout_path = values[event]
            if os.path.isfile(layout_path):
                loaded_layout = True
                marker_layout, marker_pos = load_layout(values[event])


        if event in ('Exit', None):
            break

        status, frame = cam.read()
        if status:
            # Undistort the image and display it
            if loaded_cal:
                frame = cv.remap(frame, mapx, mapy, cv.INTER_LINEAR)
                if loaded_layout:
                    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

                    draw_img = None

                    if values['draw-marker']:
                        draw_img = frame

                    ret, rvec, tvec, ids = world_pos_from_image(gray, marker_layout, camera_mtx, draw_img)

                    if ret:
                        rot_m, _ = cv.Rodrigues(rvec)
                        tvec2 = rot_m @ tvec
                        window['pos'].update(f"{np.linalg.norm(tvec2):+8.2f}cm")
                        window['x-pos'].update(f"X: {tvec[0, 0]:+8.2f}cm")
                        window['y-pos'].update(f"Y: {tvec[1, 0]:+8.2f}cm")
                        window['z-pos'].update(f"Z: {tvec[2, 0]:+8.2f}cm")

                        # Use scipy's rotation class to simplify some things
                        rot = Rotation.from_rotvec(rvec.reshape(3, ))
                        # Grab standard euler angles
                        euler = rot.as_euler('zyx', degrees=True)

                        window['x-rot'].update(f"X: {euler[2]:+8.2f}deg")
                        window['y-rot'].update(f"Y: {euler[1]:+8.2f}deg")
                        window['z-rot'].update(f"Z: {euler[0]:+8.2f}deg")
                        if values['draw-axis']:
                            for marker_id in ids:
                                rot_m, _ = cv.Rodrigues(rvec)
                                marker_tvec = marker_pos[marker_id]["pos"]
                                marker_scale = marker_pos[marker_id]["scale"]
                                draw_axis(frame, rvec, tvec + rot_m @ marker_tvec, camera_mtx, marker_scale, 2)
                else:
                    window['x-pos'].update(f"X: N/A")
                    window['y-pos'].update(f"Y: N/A")
                    window['z-pos'].update(f"Z: N/A")
                    window['x-rot'].update(f"X: N/A")
                    window['y-rot'].update(f"Y: N/A")
                    window['z-rot'].update(f"Z: N/A")

            # todo crop image to only show roi
            #  (https://stackoverflow.com/questions/39432322/what-does-the-getoptimalnewcameramatrix-do-in-opencv)
            imgbytes = cv.imencode('.png', cv.resize(frame, None, fy=0.5, fx=0.5))[1].tobytes()
            window['image'].update(data=imgbytes)

    cam.release()


if __name__ == "__main__":
    main()
