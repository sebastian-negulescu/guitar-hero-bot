import time
import cv2 as cv
import numpy as np
from threading import Event

import capture
from frame import grab_frame_from_video
import guitar

SCALE_FROM = (2560, 1440)
#SCALE_TO = (1920, 1080)
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
    (896, 1188),
    (1086, 1188),
    (1275, 1188),
    (1463, 1188),
    (1651, 1189)), SCALE_FROM, SCALE_TO)

SENSOR_BB = scale([(10, 10)], SCALE_FROM, SCALE_TO)[0]


S_MAX = 7 * 255 / 100
V_MIN = 85 * 255 / 100
V_MAX = 87 * 255 / 100


previous_detected = [False] * 5


def show_frame(frame):
    cv.imshow("frame", frame)

    key = cv.waitKey(0)
    while key != ord("q"):
        key = cv.waitKey(0)


def main():
    # fourcc = cv.VideoWriter_fourcc(*'XVID')
    # out = cv.VideoWriter('output.avi', fourcc, 30.0, SCALE_TO)
    times = open("times.txt", "w")
    g = guitar.Guitar()
    # frames = capture.ScreenCapture().generate_capture_object()
    e = Event()
    frames = grab_frame_from_video(e, "output.avi")
    # index frame with [y][x][c]
    try:
        # while True:
        for f in frames:
            start_time = time.time()
            # ret, f = frames.read()
            # out.write(f)
            # if not ret:
            #    break
            frame_hsv = cv.cvtColor(f, cv.COLOR_BGR2HSV)
            strum = False
            for lane, sensor in enumerate(SENSORS):
                # TODO: check sensor bounding box with function
                detected_note = False

                def detected_func(colour):
                    return (colour[1] <= S_MAX and
                            (V_MIN <= colour[2] and colour[2] <= V_MAX))

                detection_window = frame_hsv[SENSOR_BB[0]:SENSOR_BB[1]]

                for x in range(SENSOR_BB[0]):
                    if detected_note:
                        break

                    for y in range(SENSOR_BB[1]):
                        if detected_note:
                            break

                        pixel_colour = frame_hsv[sensor[1] + y][sensor[0] + x]
                        if (pixel_colour[1] <= S_MAX and
                                (V_MIN <= pixel_colour[2] and pixel_colour[2] <= V_MAX)):
                            detected_note = True

                if detected_note:
                    # g.press_note(lane)
                    print(lane)
                    if not previous_detected[lane]:
                        previous_detected[lane] = True
                        strum = True
                else:
                    previous_detected[lane] = False

            if strum:
                strum = False
            show_frame(f)
            end_time = time.time()
            times.write(f"{end_time - start_time}\n")
    except KeyboardInterrupt:
        pass
    finally:
        # frames.release()
        # out.release()
        times.close()


if __name__ == "__main__":
    main()
