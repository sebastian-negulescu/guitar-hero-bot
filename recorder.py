import sys
import cv2 as cv
import numpy as np

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
    output_file = sys.argv[1]

    fourcc = cv.VideoWriter_fourcc(*'FFV1')
    out = cv.VideoWriter(output_file, fourcc, 60.0, SCALE_TO)
    frames = capture.ScreenCapture().generate_capture_object()

    try:
        while True:
            ret, f = frames.read()
            if f is None:
                continue
            if not ret:
                break
            cvt_frame = cv.cvtColor(f, cv.COLOR_BGRA2BGR)
            out.write(cvt_frame)
    except KeyboardInterrupt:
        pass
    finally:
        frames.release()
        out.release()


if __name__ == "__main__":
    main()
