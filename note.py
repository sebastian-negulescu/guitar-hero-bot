import os
import pdb
import cv2 as cv
import numpy as np

from enum import Enum


TOLERANCE = 0.8
AREA_THRESHOLD = 20


class NoteColour(str, Enum):
    GREEN = 'green' 
    RED = 'red'
    YELLOW = 'yellow'
    BLUE = 'blue'
    ORANGE = 'orange' 


# formatted ((low_hsv, high_hsv), (low_hsv, high_hsv), ...)
COLOUR_RANGE_HSV = {
    NoteColour.GREEN: [[(55, 100, 100), (65, 255, 255)]],
    NoteColour.RED: [[(0, 100, 100), (5, 255, 255)], [(175, 100, 100), (179, 255, 255)]],
    NoteColour.YELLOW: [[(25, 100, 100), (35, 255, 255)]],
    NoteColour.BLUE: [[(95, 100, 100), (105, 255, 255)]],
    NoteColour.ORANGE: (),
}


class BaseNote:
    def __init__(self):
        self.template = None
        self.shape = None
        self.bounds = None


class Note:
    def __init__(self, colour, notes_dir):
        self.colour = colour
        self.notes_dir = notes_dir 

        self.base = BaseNote()
        self._load_base_note_template()


    def _load_base_note_template(self):
        note_filename = f'{self.colour}-note-base.png'
        note_path = os.path.join(self.notes_dir, note_filename)
        note_template = cv.imread(note_path)

        self.base.template = note_template
        self.base.shape = (1440, 2560)


    def find_note_base(self, image):
        '''image must be in BGR'''

        # determine the scale of the note to the image
        image_height, image_width, _ = image.shape
        
        scale_height = image_height / self.base.shape[0]
        scale_width = image_width / self.base.shape[1]

        # resize the note
        note_template = cv.resize(self.base.template, (0, 0), fx=scale_width, fy=scale_height)
        h, w = self.base.shape[0:2]

        # execute the match algorithm
        match_results = cv.matchTemplate(image, note_template, cv.TM_CCOEFF_NORMED)

        # get the best result
        _, max_val, _, max_loc = cv.minMaxLoc(match_results)
        
        if max_val < TOLERANCE:
            return False

        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        self.base.bounds = (top_left, bottom_right)
        return True


    def mask_note(self, image):
        '''image must be in HSV'''

        total_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        shades = COLOUR_RANGE_HSV[self.colour]
        for shade in shades:
            lower_colour = shade[0]
            upper_colour = shade[1]

            lower_colour_arr = np.array(lower_colour, dtype=np.uint8)
            upper_colour_arr = np.array(upper_colour, dtype=np.uint8)

            mask = cv.inRange(image, lower_colour_arr, upper_colour_arr)
            masked_img = cv.bitwise_and(image, image, mask=mask)

            total_mask = cv.bitwise_or(total_mask, masked_img)
        
        return total_mask

    @staticmethod
    def find_notes(image):
        '''image must be in grayscale'''

        _, threshold = cv.threshold(image, 1, 255, cv.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(threshold, connectivity=8)

        notes = []
        for i in range(1, num_labels):
            area = stats[i, cv.CC_STAT_AREA]
            if area < AREA_THRESHOLD:
                continue
            x, y, w, h = stats[i, cv.CC_STAT_LEFT], stats[i, cv.CC_STAT_TOP], stats[i, cv.CC_STAT_WIDTH], stats[i, cv.CC_STAT_HEIGHT]
            # TODO: filter out if area is too small

            notes.append((x, y, w, h))

        return notes

