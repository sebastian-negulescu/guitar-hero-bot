import time

import cv2 as cv
import numpy as np

from multiprocessing import Process

TICK_HZ = 30

def note_routine():
    start_time = time.time()
    print('starting')
    while time.time() < start_time + 30:
        pass
    print('finishing')
    pass


def shred():
    print('rock on!')
    p = Process(target=note_routine, args=())
    p.start()
    p.join()


if __name__ == '__main__':
    shred()
