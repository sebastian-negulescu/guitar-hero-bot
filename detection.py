import cv2 as cv
import numpy as np

from frame import grab_image, grab_frame_from_video
from threading import Event

SCALE_FROM = (2560, 1440)
SCALE_TO = (1920, 1080)


def scale(points, scale_from, scale_to):
    scaled_points = []
    for point in points:
        scaled_point = []
        for dim, p in enumerate(point):
            scaled_point.append(round(p * scale_to[dim] / scale_from[dim]))
        scaled_points.append(tuple(scaled_point))
    return scaled_points


"""
SENSORS = scale((
    (900, 1190),
    (1092, 1190),
    (1280, 1190),
    (1470, 1190),
    (1655, 1190)), SCALE_FROM, SCALE_TO)
"""

SENSORS = scale((
    (896, 1188),
    (1086, 1188),
    (1275, 1188),
    (1463, 1188),
    (1651, 1189)), SCALE_FROM, SCALE_TO)

SENSOR_BB = scale([(10, 10)], SCALE_FROM, SCALE_TO)[0]


S_MAX = 7 * 255 / 100
V_MIN = 85 * 255 / 100
V_MAX = 87 * 255 / 100


def main():
    stop_event = Event()
    frames = grab_frame_from_video(stop_event, "./testing-files/some_might_say.mkv")
    # index frame with [y][x][c]
    for f in frames:
        frame_hsv = cv.cvtColor(f, cv.COLOR_BGR2HSV)
        for sensor in SENSORS:
            """
            pixel_colour = frame_hsv[sensor[1]][sensor[0]]
            print(pixel_colour)
            if (pixel_colour[1] <= S_MAX and
                    (V_MIN <= pixel_colour[2] and pixel_colour[2] <= V_MAX)):
                print("NOTE")
                print(pixel_colour)
            """
            for x in range(SENSOR_BB[0]):
                for y in range(SENSOR_BB[1]):
                    pixel_colour = frame_hsv[sensor[1] + y][sensor[0] + x]
                    if (pixel_colour[1] <= S_MAX and
                            (V_MIN <= pixel_colour[2] and pixel_colour[2] <= V_MAX)):
                        print("NOTE")

        print()
        cv.imshow("frame", f)

        key = cv.waitKey(0)
        while key != ord("q"):
            key = cv.waitKey(0)


if __name__ == "__main__":
    main()
