import cv2
import cv2.aruco as aruco
import numpy as np
import timeit

setup = '''
import cv2

led_on_img = cv2.imread("data/led_on_mov.jpg")
led_off_img = cv2.imread("data/led_off_mov.jpg")
'''

code = '''
delta_img = cv2.absdiff(led_on_img, led_off_img)
greyscale_delta = cv2.cvtColor(delta_img, cv2.COLOR_RGB2GRAY)
_, binary_image = cv2.threshold(greyscale_delta, 127, 255, cv2.THRESH_BINARY)
'''

n = 500
run_time = timeit.timeit(setup=setup, stmt=code, number=n) / n

print(f"{run_time * 1000} ms")
