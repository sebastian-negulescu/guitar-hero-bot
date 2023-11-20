import time

# import pyautogui
import keyboard
import cv2 as cv
import numpy as np
from PIL import ImageGrab

from enum import Enum, auto
from multiprocessing import Process, Queue

MAX_COUNTER = 2**16

# REGION = (573, 911, 768, 64)
REGION = (767, 1208, 1025, 81)
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


def note_routine(colour, region, threshold, press, sync):
    print('starting', colour)
    # counter = 0

    while True:
        # lock.acquire()
        image = ImageGrab.grab(bbox=region)
        # lock.release()
        analysis_section = cv.cvtColor(np.array(image), cv.COLOR_RGB2HSV)
        # sum up the V part of the section
        accumulated_brightness = np.sum(analysis_section[::2])
        if accumulated_brightness > threshold:
            press.put(True)
        else:
            press.put(False)
        sync.get()
        '''
        counter += 1
        if counter == MAX_COUNTER:
            counter = 0  # reset the counter so we don't overflow
        '''

    print('finishing')


def shred():
    print('rock on!')

    # pillow_lock = Lock()
    note_queues = {}
    sync_queues = {}
    note_detectors = {}

    num_notes = len(NoteColours)
    for index, note_colour in enumerate(NoteColours):
        top_left_x = REGION[0] + index * REGION[2] // num_notes
        top_left_y = REGION[1]
        width = REGION[2] // num_notes
        height = REGION[3]
        # bottom_right_x = top_left_x + width
        # bottom_right_y = top_left_y + height

        # note_region = (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

        analysis_top_left_x = top_left_x + width // 2 - WINDOW[0] // 2
        analysis_top_left_y = top_left_y + height // 2 - WINDOW[1] // 2
        analysis_bottom_left_x = analysis_top_left_x + WINDOW[0]
        analysis_bottom_left_y = analysis_top_left_y + WINDOW[1]
        analysis_region = (analysis_top_left_x,
                           analysis_top_left_y,
                           analysis_bottom_left_x,
                           analysis_bottom_left_y)

        note_queues[note_colour] = Queue()
        sync_queues[note_colour] = Queue()

        note_detectors[note_colour] = Process(
            target=note_routine,
            args=(note_colour.value,
                  analysis_region,
                  DEFAULT_THRESHOLD,
                  note_queues[note_colour],
                  sync_queues[note_colour],)
        )

    for note_colour in NoteColours:
        note_detectors[note_colour].start()

    start_time = time.time()
    # counter = 0
    note_list = [''] * (len(NoteColours))
    while True:
        notes_pressed = 0
        for note_colour in NoteColours:
            press = note_queues[note_colour].get()
            sync_queues[note_colour].put(None)
            if press:
                note_list[notes_pressed] = MAPPED_KEYS[note_colour]
                notes_pressed += 1

        if notes_pressed > 0:
            for index, pressed_note in enumerate(note_list):
                if index >= notes_pressed:
                    break
                keyboard.press(pressed_note)
            keyboard.press('up')
            time.sleep(1/17)
            keyboard.release('up')
            for index, pressed_note in enumerate(note_list):
                if index >= notes_pressed:
                    break
                keyboard.release(pressed_note)

        '''
        counter += 1
        if counter == MAX_COUNTER:
            counter = 0
        '''

        end_time = time.time()
        print(f'time in milliseconds per frame: {(end_time - start_time) * 1000}')
        start_time = time.time()

    for note_colour in NoteColours:
        note_detectors[note_colour].join()


if __name__ == '__main__':
    shred()
