import cv2
import numpy as np


def process_image(path):
    img = cv2.imread(path)
    img_inv = 255 - img
    inv_gray = cv2.cvtColor(img_inv, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    img_dil = cv2.dilate(inv_gray, kernel, iterations=2)
    img_close = cv2.erode(img_dil, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        min_area_rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(min_area_rect)
        box = np.reshape(box, [-1, ])
        x1 = (box[0], box[1])
        y1 = (box[2], box[3])
        x2 = (box[4], box[5])
        y2 = (box[6], box[7])
        width = abs(x2[0] - x1[0])
        height = abs(y1[1] - y2[1])

        if width > 400 and 50 <= height <= 120:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # should be changed to crop, instead of creating
            # an image
