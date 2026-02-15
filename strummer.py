import time
import sys
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

THRESHOLD = 150

# 0 is idle
# 1 is trigger
# 2 is continue
STATE = [0, 0, 0, 0, 0]
DEBOUNCE = [0, 0, 0, 0, 0]
DEBOUNCE_DELAY = 0


def main():
    start_time = None
    end_time = None
    calibration_delay = None
    messages = []

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
            message = {"notes": []}
            for lane, sensor in enumerate(SENSORS):
                average = np.zeros(3)

                for x in range(SENSOR_BB[0]):
                    for y in range(SENSOR_BB[1]):
                        pixel_colour = frame_hsv[sensor[1] + y][sensor[0] + x]
                        average += pixel_colour

                if not end_time:
                    bottom_average = np.zeros(3)
                    for x in range(SENSOR_BB[0]):
                        for y in range(SENSOR_BB[1]):
                            pixel_colour = frame_hsv[SENSORS_BOTTOM[lane][1] + y][SENSORS_BOTTOM[lane][0] + x]
                            bottom_average += pixel_colour
                    bottom_average /= SENSOR_BB[0] * SENSOR_BB[1]
                    if bottom_average[2] >= THRESHOLD:
                        end_time = time.monotonic_ns()
                        calibration_delay = end_time - start_time

                average /= SENSOR_BB[0] * SENSOR_BB[1]
                # Four parts to a note
                passes_threshold = False
                if average[2] > 40:
                    # not crap
                    if average[1] >= 20 and average[2] >= 195:
                        passes_threshold = True
                    if average[1] >= 100:
                        passes_threshold = True
                if passes_threshold:
                    if not start_time:
                        start_time = time.monotonic_ns()

                    # good to strum
                    if STATE[lane] == 0 and time.monotonic_ns() - DEBOUNCE[lane] > DEBOUNCE_DELAY:
                        STATE[lane] = 1
                        request_to_strum = True
                        message["notes"].append(lane)
                        DEBOUNCE[lane] = time.monotonic_ns()
                else:
                    STATE[lane] = 0

            if request_to_strum:
                message["timestamp"] = time.monotonic_ns()
                messages.append(message)
                pass

            if calibration_delay is not None:
                while len(messages) > 0 and time.monotonic_ns() - messages[0]["timestamp"] > calibration_delay:
                    notes = messages.pop(0)["notes"]
                    for note in notes:
                        g.press_note(note)
                    g.strum()

    except KeyboardInterrupt:
        pass
    finally:
        frames.release()


if __name__ == "__main__":
    main()
