import os
import math
import pdb
import cv2 as cv
import numpy as np

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
                        # intersect guitar lines with base of image
                        poi_1 = point_of_intersection(line_1, [[len(image), np.pi / 2]])
                        poi_2 = point_of_intersection(line_2, [[len(image), np.pi / 2]])
                        if poi_1 is None or poi_2 is None:
                            return None
                        return [poi, poi_1, poi_2]

                    
def crop_image(image, bounds):
    # Create a mask with the same size as the image
    mask = np.zeros_like(image)

    # Create a white triangle on the mask
    triangle = np.array(bounds, dtype=np.int32)
    cv.fillPoly(mask, [triangle], (255,255,255))

    # Apply the mask to the image
    result = cv.bitwise_and(image, mask)

    # Display the cropped image
    cv.imshow('Cropped Image', result)
    cv.waitKey(0)
    cv.destroyAllWindows()


def find_notes(image, notes_path=DEFAULT_NOTES_PATH):
    # left to right
    # green, red, yellow, blue, orange
    # red note
    note_colours = [
        ('green', (0, 255, 0)), 
        ('red', (0, 0, 255)), 
        ('yellow', (0, 255, 255)), 
        ('blue', (255, 0, 0)), 
        ('orange', (0, 165, 255))
    ]
    for note_colour_name, note_colour in note_colours:
        note_filename = f'{note_colour_name}-note.png'
        note_path = os.path.join(notes_path, note_filename)
        note_template = cv.imread(note_path)
        h, w = note_template.shape[0:2]

        match_result = cv.matchTemplate(image, note_template, cv.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(match_result)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv.rectangle(image, top_left, bottom_right, note_colour, 2)

    cv.imshow('wimdow', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    screen = cv.imread('./test.png')
    guitar_points = find_guitar(screen) 
    crop_image(screen, guitar_points)
    #find_notes(screen)

if __name__ == "__main__":
    main()
