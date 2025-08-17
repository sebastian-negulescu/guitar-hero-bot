import pdb
import math
import cv2 as cv
import numpy as np

from enum import Enum
from threading import Event
from frame import grab_frame_from_video, grab_image, grab_frame

MIN_BRIGHTNESS = 150
MAX_BRIGHTNESS = 255


class Note(Enum):
    GREEN = 0
    RED = 1
    YELLOW = 2
    BLUE = 3
    ORANGE = 4


NOTE_CENTER = {
    Note.GREEN: (1050, 800),
    Note.RED: (1050, 980),
    Note.YELLOW: (1050, 1145),
    Note.BLUE: (1050, 1315),
    Note.ORANGE: (1050, 1480),
}
NOTE_BOUNDS = (4, 10)
NOTE_THRESHOLD = 0.75

SCALE_FROM = (2294, 1291)
# SCALE_FROM = (2160, 1440)
SCALE_TO = (1920, 1080)


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


def detect_note(img: cv.UMat) -> dict[Note, bool]:
    window_size = NOTE_BOUNDS[0] * NOTE_BOUNDS[1]
    has_note = {note: False for note in NOTE_CENTER.keys()}

    bounds_s = ((NOTE_BOUNDS[1] * SCALE_TO[1]) // SCALE_FROM[1], (NOTE_BOUNDS[0] * SCALE_TO[0]) // SCALE_FROM[0])

    for note, center in NOTE_CENTER.items():
        center_s = ((center[1] * SCALE_TO[1]) // SCALE_FROM[1], (center[0] * SCALE_TO[0]) // SCALE_FROM[0])
        slice_1 = (center_s[0] - bounds_s[0])
        slice_2 = (center_s[0] + bounds_s[0])
        slice_3 = (center_s[1] - bounds_s[1])
        slice_4 = (center_s[1] + bounds_s[1])
        img_crop = img[slice_1:slice_2,
                       slice_3:slice_4]
        luminance = (cv.sumElems(img_crop)[0] / window_size) / 255
        if luminance > NOTE_THRESHOLD:
            has_note[note] = True

    return has_note


def get_bar(img: cv.UMat) -> cv.UMat:
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    value_mask = cv.inRange(img_hsv, np.array([0, 0, 220]), np.array([179, 255, 255]))
    img_masked = cv.bitwise_and(img_hsv, img_hsv, mask=value_mask)
    img_colour = cv.cvtColor(img_masked, cv.COLOR_HSV2BGR)
    img_grey = cv.cvtColor(img_colour, cv.COLOR_BGR2GRAY)
    _, img_thresh = cv.threshold(img_grey, MIN_BRIGHTNESS, MAX_BRIGHTNESS, cv.THRESH_BINARY)
    return img_thresh


def shred():
    stop_event = Event()
    frames = grab_frame_from_video(stop_event, "./testing-files/guitar_hero.mkv")
    # frames = grab_image("./testing-files/reference-frame.png")
    for f in frames:
        note_detection = get_note(f)
        notes_exist = detect_note(note_detection)

        bounds_s = ((NOTE_BOUNDS[1] * SCALE_TO[1]) // SCALE_FROM[1], (NOTE_BOUNDS[0] * SCALE_TO[0]) // SCALE_FROM[0])
        note_colour = cv.cvtColor(note_detection, cv.COLOR_GRAY2BGR)

        for note, note_exists in notes_exist.items():
            center = NOTE_CENTER[note]
            center_s = ((center[1] * SCALE_TO[1]) // SCALE_FROM[1], (center[0] * SCALE_TO[0]) // SCALE_FROM[0])
            if note_exists:
                cv.line(note_colour,
                        (center_s[0] - bounds_s[0], center_s[1] - bounds_s[1]),
                        (center_s[0] - bounds_s[0], center_s[1] + bounds_s[1]),
                        (0, 0, 255), 3, cv.LINE_AA)
                cv.line(note_colour,
                        (center_s[0] - bounds_s[0], center_s[1] - bounds_s[1]),
                        (center_s[0] + bounds_s[0], center_s[1] - bounds_s[1]),
                        (0, 0, 255), 3, cv.LINE_AA)
                cv.line(note_colour,
                        (center_s[0] + bounds_s[0], center_s[1] + bounds_s[1]),
                        (center_s[0] - bounds_s[0], center_s[1] + bounds_s[1]),
                        (0, 0, 255), 3, cv.LINE_AA)
                cv.line(note_colour,
                        (center_s[0] + bounds_s[0], center_s[1] + bounds_s[1]),
                        (center_s[0] + bounds_s[0], center_s[1] - bounds_s[1]),
                        (0, 0, 255), 3, cv.LINE_AA)

        cv.imshow("frame", note_colour)

        key = cv.waitKey(0)
        if key == ord("q"):
            stop_event.set()

    cv.destroyAllWindows()


if __name__ == "__main__":
    shred()
