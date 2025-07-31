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
    src = cv.imread(cv.samples.findFile(image_path), cv.IMREAD_GRAYSCALE)
    assert src is not None
    return [src]  # for generator compatibility
