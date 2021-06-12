import cv2

led_on_img = cv2.imread("data/led_on.jpg")
led_off_img = cv2.imread("data/led_off.jpg")

delta_img = cv2.absdiff(led_on_img, led_off_img)
greyscale_delta = cv2.cvtColor(delta_img, cv2.COLOR_RGB2GRAY)
_, binary_image = cv2.threshold(greyscale_delta, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite("output/delta_grey.png", greyscale_delta)
cv2.imwrite("output/delta.png", delta_img)
cv2.imwrite("output/binary.png", binary_image)
cv2.waitKey(0)