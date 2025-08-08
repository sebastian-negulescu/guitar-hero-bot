import math
import cv2 as cv
import numpy as np

from enum import Enum
from frame import grab_image

MIN_BRIGHTNESS = 150
MAX_BRIGHTNESS = 255


class State(Enum):
    NECK = 0
    NOTES = 1


def get_lines(img: cv.UMat) -> cv.UMat:
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    _, img_thresh = cv.threshold(img_gray, MIN_BRIGHTNESS, MAX_BRIGHTNESS, cv.THRESH_BINARY)

    LOWER_THRESHOLD = 100
    HIGHER_THRESHOLD = 150
    img_edges = cv.Canny(img_thresh, LOWER_THRESHOLD, HIGHER_THRESHOLD,
                         edges=None, L2gradient=3)

    KERNEL_DIM = (5, 5)
    kernel_mat = np.ones(KERNEL_DIM, np.float32) / (KERNEL_DIM[0] * KERNEL_DIM[1])
    img_smoothed = cv.filter2D(img_edges, -1, kernel_mat)

    RHO = 1
    THETA = np.pi / 180
    THRESHOLD = 50
    lines = cv.HoughLinesP(img_smoothed, RHO, THETA, THRESHOLD,
                           lines=None, minLineLength=250, maxLineGap=2)

    return lines


def find_neck(img: cv.UMat):
    lines = get_lines(img)
    if lines is not None:
        for line in lines:
            pass


def get_note(img: cv.UMat) -> cv.UMat:
    white_mask = cv.inRange(img, np.array([210, 210, 210]), np.array([255, 255, 255]))
    img_masked = cv.bitwise_and(img, img, mask=white_mask)
    img_grey = cv.cvtColor(img_masked, cv.COLOR_BGR2GRAY)
    _, img_thresh = cv.threshold(img_grey, MIN_BRIGHTNESS, MAX_BRIGHTNESS, cv.THRESH_BINARY)
    return img_thresh


def get_bar(img: cv.UMat) -> cv.UMat:
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    value_mask = cv.inRange(img_hsv, np.array([0, 0, 220]), np.array([179, 255, 255]))
    img_masked = cv.bitwise_and(img_hsv, img_hsv, mask=value_mask)
    img_colour = cv.cvtColor(img_masked, cv.COLOR_HSV2BGR)
    img_grey = cv.cvtColor(img_colour, cv.COLOR_BGR2GRAY)
    _, img_thresh = cv.threshold(img_grey, MIN_BRIGHTNESS, MAX_BRIGHTNESS, cv.THRESH_BINARY)
    return img_thresh


def shred():
    for f in grab_image("./testing-files/reference-frame.png"):
        note_detection = get_note(f)


    key = cv.waitKey(0)
    while key != ord("q"):
        key = cv.waitKey(0)
    cv.destroyAllWindows()



if __name__ == "__main__":
    shred()
