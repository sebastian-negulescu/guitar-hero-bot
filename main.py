import time

import cv2 as cv
import numpy as np

from enum import Enum

from multiprocessing import Process

TICK_HZ = 30

class NoteColours(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2


def note_routine():
    start_time = time.time()
    print('starting')
    while time.time() < start_time + 30:
        pass
    print('finishing')
    pass


def shred():
    print('rock on!')

    note_detectors = {}
    for note_colour in NoteColours:
        note_detectors[note_colour] = Process(target=note_routine)

    for note_colour in NoteColours:
        note_detectors[note_colour].start()

    for note_colour in NoteColours:
        note_detectors[note_colour].join()


if __name__ == '__main__':
    shred()
