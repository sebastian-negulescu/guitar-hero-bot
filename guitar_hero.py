import math
import cv2 as cv
import numpy as np

from enum import Enum
from frame import grab_image


class State(Enum):
    NECK = 0
    NOTES = 1


def find_neck(src: cv.UMat):
    ret, thresh = cv.threshold(src, 150, 255, cv.THRESH_BINARY)

    """
    original_params = (len(dst[0]), len(dst))
    resize_params = (len(dst[0]) // 2, len(dst) // 2)
    # dst_r = cv.resize(dst, resize_params, interpolation=cv.INTER_LINEAR)
    dst_r = dst
    """

    """
    kernel = np.ones((5, 5), np.float32) / 25
    smoothed = cv.filter2D(thresh, -1, kernel)
    """

    dst = cv.Canny(thresh, 100, 150, None, 3)

    kernel = np.ones((5, 5), np.float32) / 25
    smoothed = cv.filter2D(dst, -1, kernel)

    ret, thresh = cv.threshold(src, 150, 255, cv.THRESH_BINARY)

    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    # lines = cv.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)

    lines = None
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0, 0, 255), 3, cv.LINE_AA)

    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 250, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Source", smoothed)
    # cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
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
