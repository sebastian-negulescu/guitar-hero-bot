import math
import cv2 as cv
import numpy as np

from enum import Enum
from frame import grab_image


class State(Enum):
    NECK = 0
    NOTES = 1


def find_neck(img: cv.UMat):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    MIN_BRIGHTNESS = 150
    MAX_BRIGHTNESS = 255
    _, img_thresh = cv.threshold(img_gray, MIN_BRIGHTNESS, MAX_BRIGHTNESS, cv.THRESH_BINARY)

    LOWER_THRESHOLD = 100
    HIGHER_THRESHOLD = 150
    img_edges = cv.Canny(img_thresh, LOWER_THRESHOLD, HIGHER_THRESHOLD,
                         edges=None, L2gradient=3)

    KERNEL_DIM = (5, 5)
    kernel_mat = np.ones(KERNEL_DIM, np.float32) / (KERNEL_DIM[0] * KERNEL_DIM[1])
    img_smoothed = cv.filter2D(img_edges, -1, kernel_mat)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(img_smoothed, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    RHO = 1
    THETA = np.pi / 180
    THRESHOLD = 50
    lines = cv.HoughLinesP(img_smoothed, RHO, THETA, THRESHOLD,
                           lines=None, minLineLength=250, maxLineGap=2)

    if lines is not None:
        for i in range(0, len(lines)):
            line = lines[i][0]
            cv.line(cdstP, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Source", img_smoothed)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    cv.waitKey()
    return 0


def shred():
    state = State.NECK
    for f in grab_image("./testing-files/reference-frame.png"):
        match state:
            case State.NECK:
                find_neck(f)


if __name__ == "__main__":
    shred()
