import time

import cv2 as cv
import numpy as np
from PIL import ImageGrab

from enum import Enum, auto
from multiprocessing import Process, Queue

TICK_HZ = 5
TIME_PER_FRAME = 1 / TICK_HZ
QUIT = False
REGION = (573, 911, 768, 64)
WINDOW = (10, 10)
DEFAULT_THRESHOLD = 60000

class NoteColours(Enum):
    GREEN = auto() 
    RED = auto()
    YELLOW = auto()
    BLUE = auto()
    ORANGE = auto()


class TimeStamped:
    def __init__(self, obj, terminate=False):
        self.obj = obj
        self.terminate = terminate
        self.timestamp = time.time()


def note_routine(colour, frames, threshold):
    print('starting')
    timestamped_obj = None
    # signal_func = []
    while True: 
        # check queue
        try:
            timestamped_obj = frames.get()
            if timestamped_obj.terminate:
                break
        except ValueError:
            # queue is closed
            break

        frame = timestamped_obj.obj
        half = (len(frame) // 2, len(frame[0]) // 2)
        start_range = (half[0] - WINDOW[0] // 2, half[1] - WINDOW[1] // 2)
        analysis_area = frame[start_range[0]:start_range[0] + WINDOW[0], 
                              start_range[1]:start_range[1] + WINDOW[1]]
        accumulated_brightness = np.sum(analysis_area[::2])
        # signal_func.append(accumulated_brightness)

        if accumulated_brightness > threshold:
            pass # we got em

        # check if the time is still valid
    # print(signal_func)
    print('finishing')


def shred():
    print('rock on!')

    note_queues = {}
    note_detectors = {}

    for note_colour in NoteColours:
        note_queues[note_colour] = Queue()
        note_detectors[note_colour] = Process(
            target=note_routine, 
            args=(note_colour.value, note_queues[note_colour], DEFAULT_THRESHOLD,)
        )

    for note_colour in NoteColours:
        note_detectors[note_colour].start()

    ''' replace with a video capture while figuring things out
    while not QUIT:
        frame_timestamp = time.time()
        frame = ImageGrab.grab(REGION)

        for note_colour in NoteColours:
            note_queues[note_colour].put(TimeStamped(frame, frame_timestamp))

        end_time = time.time()
        if TIME_PER_FRAME > end_time - frame_timestamp:
            time.sleep(TIME_PER_FRAME - (end_time - frame_timestamp))
    '''

    video = cv.VideoCapture('testing-files/test-movie.mp4')
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        cropped_frame = frame[REGION[1]:REGION[1] + REGION[3], 
                              REGION[0]:REGION[0] + REGION[2]]
        cropped_frame_hsv = cv.cvtColor(cropped_frame, cv.COLOR_BGR2HSV)

        for index, note_colour in enumerate(NoteColours):
            start_range = REGION[2] * index // len(NoteColours)
            end_range = REGION[2] * (index + 1) // len(NoteColours)
            note_frame = cropped_frame_hsv[:, start_range:end_range]
            note_queues[note_colour].put(TimeStamped(note_frame))

        cv.imshow('frame', cropped_frame)

        if cv.waitKey(1) == ord('q'):
            break

    for note_colour in NoteColours:
        note_queues[note_colour].put(TimeStamped(None, terminate=True))

    for note_colour in NoteColours:
        note_detectors[note_colour].join()


if __name__ == '__main__':
    shred()
