import cv2
import numpy as np

from threading import Event
from frame import grab_frame


def shred():
    stop_event = Event()
    for f in grab_frame(stop_event):
        pass


if __name__ == "__main__":
    shred()
