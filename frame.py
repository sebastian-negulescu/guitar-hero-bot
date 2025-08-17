import mss
import cv2 as cv

from threading import Event
from typing import Generator


def grab_frame(
    stop_event: Event
) -> Generator[mss.base.ScreenShot, None, None]:
    with mss.mss() as sct:
        while not stop_event.is_set():
            yield sct.grab()


def grab_image(image_path: str) -> list[cv.UMat]:
    src = cv.imread(cv.samples.findFile(image_path))
    assert src is not None
    return [src]  # for generator compatibility


def grab_frame_from_video(
    stop_event: Event,
    video_path: str
) -> Generator[cv.UMat, None, None]:
    capture = cv.VideoCapture(video_path)
    while not stop_event.is_set():
        ret, src = capture.read()
        if not ret:
            break
        yield src
    capture.release()
