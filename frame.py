import mss
from threading import Event
from typing import Generator


def grab_frame(
    stop_event: Event
) -> Generator[mss.base.ScreenShot, None, None]:
    with mss.mss() as sct:
        while not stop_event.is_set():
            yield sct.grab()
