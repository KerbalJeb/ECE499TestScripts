import argparse
from PySimpleGUI.PySimpleGUI import Window

import cv2 as cv
import numpy as np
import PySimpleGUI as sg
import os.path
import re

from scipy.spatial.transform import Rotation

from drawing_util import draw_axis
from tracking_util import load_cal_data, world_pos_from_image, load_layout

EventFunctions = {}

W = 1920
H = 1080
dim = (W, H)


def main():
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=int, default=0, help="The id of the camera to use")
    parser.add_argument("--cal", type=str, help="The path to the npa camera calibration data")
    parser.add_argument("--layout", type=str, help='The marker layout json file')

    args = parser.parse_args()

    loaded_cal = False
    loaded_layout = False

    # Setup the GUI
    sg.theme('Black')

    # todo add text input for target position and required accuracy

    # define the window layout

    control_panel = [
        [sg.Checkbox('Draw Axis', key='draw-axis', default=False)],
        [sg.Checkbox('Draw Marker', key='draw-marker', default=False)],
        [sg.Checkbox('Draw Origin', key='draw-origin', default=False)],
    ]

    target_input = [
        [sg.Input(key='x-target', enable_events=True, size=(8, 1), default_text='0.0')],
        [sg.Input(key='y-target', enable_events=True, size=(8, 1), default_text='0.0')],
        [sg.Input(key='z-target', enable_events=True, size=(8, 1), default_text='0.0')],
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
        [sg.Column([[sg.Frame('Control Panel', layout=control_panel, vertical_alignment='top')],
                    [sg.Frame('Target', layout=target_input)],
                    [sg.Image(filename='', key='widget')]],
                   vertical_alignment='top'),
         sg.Image(filename='', key='image'),
         sg.Column([
             [sg.Frame('Camera-Space Position', vertical_alignment='top', layout=[
                 [sg.Text('N/A', key='pos', size=(15, 1))],
                 [sg.Text('X: N/A', key='x-pos', size=(15, 1))],
                 [sg.Text('Y: N/A', key='y-pos', size=(15, 1))],
                 [sg.Text('Z: N/A', key='z-pos', size=(15, 1))],
             ])],
             [sg.Frame('World-Space Position', vertical_alignment='top', layout=[
                 [sg.Text('N/A', key='pos-world', size=(15, 1))],
                 [sg.Text('X: N/A', key='x-pos-world', size=(15, 1))],
                 [sg.Text('Y: N/A', key='y-pos-world', size=(15, 1))],
                 [sg.Text('Z: N/A', key='z-pos-world', size=(15, 1))],
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
    _, _ = window.read(timeout=0)

    if args.cal:
        if not os.path.isfile(args.cal):
            print(f"{args} does not exits")
        window.write_event_value('cal-path', args.cal)
        window["cal-path"].update(args.cal)

    if args.layout:
        if not os.path.isfile(args.layout):
            print(f"{args} does not exits")
        window.write_event_value('layout-path', args.layout)
        window["layout-path"].update(args.layout)

    # Get the camera
    camera_id = args.camera
    cam = cv.VideoCapture(camera_id, cv.CAP_DSHOW)

    if not cam.isOpened():
        print("Failed to load camera")
        return

    cam.set(cv.CAP_PROP_FRAME_WIDTH, W)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, H)
    widget_img = 255 * np.ones((250, 250, 3), dtype=np.uint8)
    target_pos = np.zeros((3, 1))

    # Event loop
    while True:
        # todo call functions to find camera pos from markers
        # todo update GUI as needed
        event, values = window.read(timeout=20)

        if event == 'x-target':
            text = values[event]
            if not re.match(r"^(?:\d+(?:\.\d*)?|\.\d+)$", text):
                window[event].update(text[:-1])
            target_pos[0, 0] = float(text)

        if event == 'y-target':
            text = values[event]
            if not re.match(r"^(?:\d+(?:\.\d*)?|\.\d+)$", text):
                window[event].update(text[:-1])
            target_pos[1, 0] = float(text)

        if event == 'z-target':
            text = values[event]
            if not re.match(r"^(?:\d+(?:\.\d*)?|\.\d+)$", text):
                window[event].update(text[:-1])
            target_pos[2, 0] = float(text)

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
                widget_img.fill(255)
                if loaded_layout:
                    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

                    draw_img = None

                    if values['draw-marker']:
                        draw_img = frame

                    ret, rvec, tvec, ids = world_pos_from_image(gray, marker_layout, camera_mtx, draw_img)

                    if ret:
                        Rt, _ = cv.Rodrigues(rvec)
                        R = Rt.transpose()
                        pos = -R @ tvec
                        window['pos'].update(f"{np.linalg.norm(tvec):+8.2f}cm")
                        window['x-pos'].update(f"X: {tvec[0, 0]:+8.2f}cm")
                        window['y-pos'].update(f"Y: {tvec[1, 0]:+8.2f}cm")
                        window['z-pos'].update(f"Z: {tvec[2, 0]:+8.2f}cm")

                        window['pos-world'].update(f"{np.linalg.norm(pos):+8.2f}cm")
                        window['x-pos-world'].update(f"X: {pos[0, 0]:+8.2f}cm")
                        window['y-pos-world'].update(f"Y: {pos[1, 0]:+8.2f}cm")
                        window['z-pos-world'].update(f"Z: {pos[2, 0]:+8.2f}cm")

                        terror = target_pos - tvec
                        xPos = int(np.clip(terror[0, 0], -100, 100))
                        xPos += 125
                        yPos = int(np.clip(terror[1, 0], -100, 100))
                        yPos += 125
                        zPos = int(np.clip(terror[2, 0], -100, 100))
                        zPos += 125

                        cv.circle(widget_img, (xPos, yPos), 15, color=(0, 0, 255), thickness=2)
                        cv.line(widget_img, (125, 0), (125, 250), color=(0, 0, 0), thickness=1)
                        cv.line(widget_img, (0, 125), (250, 125), color=(0, 0, 0), thickness=1)
                        cv.line(widget_img, (5, zPos), (20, zPos), color=(255, 0, 0), thickness=2)

                        # Use scipy's rotation class to simplify some things
                        rot = Rotation.from_matrix(Rt)
                        # Grab standard euler angles
                        euler = rot.as_euler('xyz', degrees=True)

                        window['x-rot'].update(f"X: {euler[0]:+8.2f}deg")
                        window['y-rot'].update(f"Y: {euler[1]:+8.2f}deg")
                        window['z-rot'].update(f"Z: {euler[2]:+8.2f}deg")
                        if values['draw-axis']:
                            for marker_id in ids:
                                marker_tvec = marker_pos[marker_id]["pos"]
                                marker_scale = marker_pos[marker_id]["scale"]
                                draw_axis(frame, rvec, tvec + Rt @ marker_tvec, camera_mtx, marker_scale, 2)
                        if values['draw-origin']:
                            draw_axis(frame, rvec, tvec, camera_mtx, 25, 2)
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
            imgbytes = cv.imencode('.png', widget_img)[1].tobytes()
            window['widget'].update(data=imgbytes)

    cam.release()


if __name__ == "__main__":
    main()
