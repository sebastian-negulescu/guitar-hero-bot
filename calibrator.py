import sys
import cv2 as cv
import numpy as np
from threading import Event

from frame import grab_frame_from_video

SCALE_FROM = (2560, 1440)
SCALE_TO = SCALE_FROM


def scale(points, scale_from, scale_to):
    scaled_points = []
    for point in points:
        scaled_point = []
        for dim, p in enumerate(point):
            scaled_point.append(round(p * scale_to[dim] / scale_from[dim]))
        scaled_points.append(tuple(scaled_point))
    return scaled_points


SENSORS = scale((
    (1071, 824),
    (1175, 824),
    (1280, 824),
    (1384, 824),
    (1489, 824)), SCALE_FROM, SCALE_TO)

SENSORS_BOTTOM = scale((
    (899, 1190),
    (1057, 1207),
    (1275, 1189),
    (1457, 1207),
    (1643, 1207)), SCALE_FROM, SCALE_TO)

SENSOR_BB = scale([(1, 1)], SCALE_FROM, SCALE_TO)[0]


def show_frame(frame):
    cv.imshow("frame", frame)

    key = cv.waitKey(0)
    while key != ord("q"):
        key = cv.waitKey(0)


def main():
    input_file = sys.argv[1]

    e = Event()
    frames = grab_frame_from_video(e, input_file)
    # index frame with [y][x][c]
    try:
        for f in frames:
            frame_hsv = cv.cvtColor(f, cv.COLOR_BGR2HSV)
            print()
            for lane, sensor in enumerate(SENSORS):
                average = np.zeros(3)

                for x in range(SENSOR_BB[0]):
                    for y in range(SENSOR_BB[1]):
                        pixel_colour = frame_hsv[sensor[1] + y][sensor[0] + x]
                        average += pixel_colour

                average /= SENSOR_BB[0] * SENSOR_BB[1]
                print(lane, average)

            show_frame(f)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
