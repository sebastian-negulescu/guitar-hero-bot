import time
import signal
import sys

import cv2 as cv
import numpy as np
from PIL import ImageGrab

from enum import Enum, auto
from multiprocessing import Process, Queue

TICK_HZ = 5
TIME_PER_FRAME = 1 / TICK_HZ
QUIT = False

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


def capture_interrupt(sig, frame):
    # this will run on all the threads
    QUIT = True
    sys.exit(0)

def shred():
    print('rock on!')

    note_queues = {}
    note_detectors = {}

    signal.signal(signal.SIGINT, capture_interrupt)

    for note_colour in NoteColours:
        note_queues[note_colour] = Queue()
        note_detectors[note_colour] = Process(
            target=note_routine, 
            args=(note_queues[note_colour],)
        )

    for note_colour in NoteColours:
        note_detectors[note_colour].start()

    while not QUIT:
        frame_timestamp = time.time()
        frame = ImageGrab.grab()

        for note_colour in NoteColours:
            note_queues[note_colour].put(TimeStamped(frame, frame_timestamp))

        end_time = time.time()
        if TIME_PER_FRAME < end_time - frame_timestamp:
            time.sleep(TIME_PER_FRAME - (end_time - frame_timestamp))

    print('cleaning up')
    for note_colour in NoteColours:
        note_queues[note_colour].put(TimeStamped('quit'))

    for note_colour in NoteColours:
        note_detectors[note_colour].join()


if __name__ == '__main__':
    shred()
