from PIL import ImageGrab
import cv2 as cv
import numpy as np

from threading import Event
from typing import Generator, Any


def grab_frame(
    stop_event: Event
) -> Generator[Any, None, None]:
    while not stop_event.is_set():
        im = ImageGrab.grab()
        yield cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)


def grab_image(image_path: str) -> list[cv.UMat]:
    src = cv.imread(cv.samples.findFile(image_path))
    assert src is not None
    return [src, src, src]  # for generator compatibility


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
