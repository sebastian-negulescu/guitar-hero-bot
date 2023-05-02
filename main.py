import os
import math
import time
import pdb
import cv2 as cv
import numpy as np

from note import Note, NoteColour

PATH = os.path.dirname(os.path.realpath(__file__))
DEFAULT_NOTES_PATH = os.path.join(PATH, './notes')

EPSILON = 1e-5
MIDDLE_TOLERANCE = 5


def point_of_intersection(a, b):
    (radius_a, theta_a) = a[0]
    (radius_b, theta_b) = b[0]

    sin_a = math.sin(theta_a)
    sin_b = math.sin(theta_b)

    cos_a = math.cos(theta_a)
    cos_b = math.cos(theta_b)

    if math.isclose(sin_a, 0, rel_tol=EPSILON):
        return None

    if math.isclose(sin_b, 0, rel_tol=EPSILON):
        return None

    part_a = cos_a / sin_a
    part_b = cos_b / sin_b

    if math.isclose(part_a, part_b, rel_tol=EPSILON):
        return None

    x = ((radius_a/sin_a - radius_b/sin_b) / (part_a - part_b))
    y = -part_a * x + radius_a/sin_a
    return (x, y)


def find_guitar(image):
    edges = cv.Canny(image, 50, 200, None, 3)
    lines = cv.HoughLines(edges, 1, np.pi / 180, 400, None, 0, 0)

    filtered_lines = []
    # draw the lines
    if lines is not None:
        for line in lines:
            theta = line[0][1]
            # np.pi / 2 is horizontal line, 0 is vertical
            if not (math.isclose(theta, np.pi / 2, rel_tol=EPSILON) or 
                    math.isclose(theta, 0, rel_tol=EPSILON)): # catch horizontal and vertical lines
                filtered_lines.append(line)

    if len(filtered_lines) > 0: 
        for line_1 in filtered_lines:
            theta_1 = line_1[0][1]
            for line_2 in filtered_lines:
                theta_2 = line_2[0][1]
                
                angles_opposite = math.isclose(abs(np.pi - theta_1), theta_2, rel_tol=EPSILON)

                if angles_opposite:
                    poi = point_of_intersection(line_1, line_2)
                    if poi is None:
                        continue
                    cols = image.shape[1]

                    if math.isclose(poi[0], cols, rel_tol=MIDDLE_TOLERANCE):
                        # intersect guitar lines with base of image to form a triangle
                        poi_1 = point_of_intersection(line_1, [[len(image), np.pi / 2]])
                        poi_2 = point_of_intersection(line_2, [[len(image), np.pi / 2]])
                        if poi_1 is None or poi_2 is None:
                            return None
                        # return the 3 points forming the triangle
                        return [poi, poi_1, poi_2]

 
def crop_image(image, bounds):
    # create a mask with the same size as the image
    mask = np.zeros_like(image)

    # create a white triangle on the mask
    triangle = np.array(bounds, dtype=np.int32)
    cv.fillPoly(mask, [triangle], (255,255,255))

    # apply the mask to the image
    result = cv.bitwise_and(image, mask)

    return result


def main():
    screen = cv.imread('./test-multiple-notes.png')

    notes = {
        NoteColour.GREEN: None,
        NoteColour.RED: None,
        NoteColour.YELLOW: None,
        NoteColour.BLUE: None,
        NoteColour.ORANGE: None,
    }
    for note_colour in notes.keys():
        notes[note_colour] = Note(note_colour, DEFAULT_NOTES_PATH)

    guitar_points = find_guitar(screen) 
    guitar_cropped_image = crop_image(screen, guitar_points)
    guitar_cropped_image_hsv = cv.cvtColor(guitar_cropped_image, cv.COLOR_BGR2HSV)

    masked_notes_image = np.zeros((screen.shape[0], screen.shape[1], 3), dtype=np.uint8)
    for note in notes.values():
        note.find_note_base(guitar_cropped_image)
        masked_note_image = note.mask_note(guitar_cropped_image_hsv)
        masked_notes_image = cv.bitwise_or(masked_notes_image, masked_note_image)

    _, _, masked_notes_image_grey = cv.split(masked_notes_image)

    notes_coords = Note.find_notes(masked_notes_image_grey)
    # TODO: check if notes are within the bounds 

if __name__ == "__main__":
    main()
