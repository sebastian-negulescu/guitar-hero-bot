import evdev

import cv2 as cv
import numpy as np

import guitar
import capture

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
    (899, 1190),
    (1087, 1207),
    (1281, 1189),
    (1457, 1207),
    (1643, 1207)), SCALE_FROM, SCALE_TO)

SENSOR_BB = scale([(1, 1)], SCALE_FROM, SCALE_TO)[0]

THRESHOLDS = [100, 100, 100, 100, 100]

# 0 is idle
# 1 is trigger
# 2 is continue
STATE = [0, 0, 0, 0, 0]


def main():
    device = evdev.InputDevice("/dev/input/event4")
    activated = False

    g = guitar.Guitar()

    frames = capture.ScreenCapture().generate_capture_object()
    # index frame with [y][x][c]
    try:
        while True:
            ret, f = frames.read()
            if f is None:
                continue
            if not ret:
                break
            frame_hsv = cv.cvtColor(f, cv.COLOR_BGR2HSV)

            if 16 in device.active_keys():
                activated = True
            if 17 in device.active_keys():
                activated = False
            if not activated:
                continue

            request_to_strum = False
            for lane, sensor in enumerate(SENSORS):
                average = np.zeros(3)

                for x in range(SENSOR_BB[0]):
                    for y in range(SENSOR_BB[1]):
                        pixel_colour = frame_hsv[sensor[1] + y][sensor[0] + x]
                        average += pixel_colour

                average /= SENSOR_BB[0] * SENSOR_BB[1]
                # print(average)
                press = False
                release = False
                if average[2] >= THRESHOLDS[lane]:
                    # good to strum
                    if STATE[lane] == 0:
                        STATE[lane] = 1
                        request_to_strum = True
                        press = True
                elif STATE[lane] > 0:
                    STATE[lane] = 0
                    release = True

                if press:
                    g.press_note(lane)
                elif release:
                    pass

            if request_to_strum:
                print("strum")
                g.strum()
    except KeyboardInterrupt:
        pass
    finally:
        frames.release()


if __name__ == "__main__":
    main()
