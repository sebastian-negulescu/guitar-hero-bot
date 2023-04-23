import os
import math
import time
import cv2 as cv
import numpy as np

from note import Note, NoteType

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

    # display the cropped image
    cv.imshow('guitar', result)
    cv.waitKey(0)
    cv.destroyAllWindows()


def load_notes(type, notes_path=DEFAULT_NOTES_PATH):
    notes = {}
    note_colours = [
        ('green', (0, 255, 0)), 
        ('red', (0, 0, 255)), 
        ('yellow', (0, 255, 255)), 
        ('blue', (255, 0, 0)), 
        ('orange', (0, 165, 255))
    ]
    for note_colour_name, note_colour in note_colours:
        note_filename = None
        if type == NoteType.regular:
            note_filename = f'{note_colour_name}-note.png'
        if type == NoteType.base:
            note_filename = f'{note_colour_name}-note-base.png'
        note_path = os.path.join(notes_path, note_filename)
        note_template = cv.imread(note_path)
        notes[note_colour_name] = Note(note_template, note_colour, (1440, 2560))

    return notes


def find_note_bases(image, notes):
    image_height, image_width, _ = image.shape
    for note in notes.values():
        # compute the scale relative to the note
        scale_width = image_width / note.shape[1]
        scale_height = image_height / note.shape[0]

        # resize the note
        note_template = cv.resize(note.template, (0, 0), fx=scale_width, fy=scale_height)
        h, w = note_template.shape[0:2]

        # execute the match algorithm
        match_results = cv.matchTemplate(image, note_template, cv.TM_CCOEFF_NORMED)

        # get the best result
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match_results)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        note.bounds = (top_left, bottom_right)

        cv.rectangle(image, top_left, bottom_right, note.colour, 2)

    return notes


def find_notes(image, notes):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    yellow_low = (25, 100, 100)
    yellow_high = (35, 255, 255)

    red_low = (170, 50, 50)
    red_high = (179, 255, 255)

    green_low = (55, 50, 100)
    green_high = (65, 255, 255)
    
    lower_white = np.array(green_low, dtype=np.uint8)
    upper_white = np.array(green_high, dtype=np.uint8)

    mask = cv.inRange(hsv, lower_white, upper_white)
    masked_img = cv.bitwise_and(image, image, mask=mask)

    cv.imshow('masked', masked_img)
    cv.waitKey(0)

    # now we perform connected component analysis
    gray = cv.cvtColor(cv.cvtColor(masked_img, cv.COLOR_HSV2BGR), cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity=8)

    output = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8) 

    for i in range(1, nlabels):
        # Get the area of the connected component
        area = stats[i, cv.CC_STAT_AREA]
        # Get the bounding box coordinates of the connected component
        x, y, w, h = stats[i, cv.CC_STAT_LEFT], stats[i, cv.CC_STAT_TOP], stats[i, cv.CC_STAT_WIDTH], stats[i, cv.CC_STAT_HEIGHT]
        # Draw a rectangle around the connected component
        cv.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Print the area of the connected component
        print("Connected component %d has area %d" % (i, area))

    cv.imshow('connected', output)

    
def main():
    screen = cv.imread('./test-multiple-notes.png')
    guitar_points = find_guitar(screen) 
    # crop_image(screen, guitar_points)
    note_bases = load_notes(NoteType.base)
    notes = load_notes(NoteType.regular)
    
    start = time.time()
    # find_note_bases(screen, note_bases)
    find_notes(screen, notes)
    end = time.time()
    print(end - start)

    #cv.imshow('blep', screen)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
