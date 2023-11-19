import time

import pyautogui
import cv2 as cv
import numpy as np
from PIL import ImageGrab

from enum import Enum, auto
from multiprocessing import Process, Queue, Lock

MAX_COUNTER = 2**16

REGION = (573, 911, 768, 64)
# REGION = (767, 1208, 1025, 81)
WINDOW = (20, 20)
DEFAULT_THRESHOLD = 150 * WINDOW[0] * WINDOW[1]

class NoteColours(Enum):
    GREEN = auto() 
    RED = auto()
    YELLOW = auto()
    BLUE = auto()
    ORANGE = auto()


MAPPED_KEYS = {
    NoteColours.GREEN: 'q',
    NoteColours.RED: 'w',
    NoteColours.YELLOW: 'e',
    NoteColours.BLUE: 'r',
    NoteColours.ORANGE: 't'
}

def note_routine(colour, region, threshold, press, lock):
    print('starting', colour)
    counter = 0

    while True:
        # lock.acquire()
        image = ImageGrab.grab(bbox=region)
        # lock.release()
        analysis_section = cv.cvtColor(np.array(image), cv.COLOR_RGB2HSV)
        accumulated_brightness = np.sum(analysis_section[::2]) # sum up the V part of the section  

        if accumulated_brightness > threshold:
            press.put((True, counter))
        else:
            press.put((False, counter))
        counter += 1
        if counter == MAX_COUNTER:
            counter = 0 # reset the counter so we don't overflow

    print('finishing')


def shred():
    print('rock on!')

    pillow_lock = Lock()
    note_queues = {}
    note_detectors = {}

    num_notes = len(NoteColours)
    for index, note_colour in enumerate(NoteColours):
        top_left_x = REGION[0] + index * REGION[2] // num_notes
        top_left_y = REGION[1]
        width = REGION[2] // num_notes
        height = REGION[3]
        bottom_right_x = top_left_x + width
        bottom_right_y = top_left_y + height

        note_region = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        note_queues[note_colour] = Queue()

        note_detectors[note_colour] = Process(
            target=note_routine, 
            args=(note_colour.value, 
                  note_region,
                  DEFAULT_THRESHOLD, 
                  note_queues[note_colour],
                  pillow_lock,)
        )

    for note_colour in NoteColours:
        note_detectors[note_colour].start()

    counter = 0
    while True:
        for note_colour in NoteColours:
            press, note_count = note_queues[note_colour].get()

        print(counter)
        counter += 1
        if counter == MAX_COUNTER:
            counter = 0
        
    for note_colour in NoteColours:
        note_detectors[note_colour].join()


if __name__ == '__main__':
    shred()
