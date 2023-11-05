import time

import cv2 as cv
import numpy as np
from PIL import ImageGrab

from enum import Enum, auto
from multiprocessing import Process, Queue

TICK_HZ = 30
QUIT_VALUE = 113

class NoteColours(Enum):
    GREEN = auto() 
    RED = auto()
    YELLOW = auto()
    BLUE = auto()
    ORANGE = auto()


class TimeStamped:
    def __init__(self, obj, curr_time = time.time()):
        self.obj = obj
        self.timestamp = curr_time


def note_routine(frames):
    print('starting')
    while True: 
        # check queue
        try:
            timestamped_obj = frames.get()
            if timestamped_obj.obj == 'quit':
                break
        except ValueError:
            # queue is closed
            break

        # check if the time is still valid
    print('finishing')


def shred():
    print('rock on!')

    note_queues = {}
    note_detectors = {}
    for note_colour in NoteColours:
        note_queues[note_colour] = Queue()
        note_detectors[note_colour] = Process(
            target=note_routine, 
            args=(note_queues[note_colour],)
        )

    for note_colour in NoteColours:
        note_detectors[note_colour].start()

    while True:
        frame_time = time.time()
        frame = ImageGrab.grab()
        print(time.time() - frame_time)
        for note_colour in NoteColours:
            pass

        if cv.waitKey(1) == QUIT_VALUE:
            break

    for note_colour in NoteColours:
        note_queues[note_colour].put(TimeStamped('quit'))

    for note_colour in NoteColours:
        note_detectors[note_colour].join()


if __name__ == '__main__':
    shred()
