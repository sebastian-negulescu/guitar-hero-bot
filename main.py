import math
import cv2 as cv
import numpy as np

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

    # Copy edges to the images that will display the results in BGR
    drawn_outline = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    lines = cv.HoughLines(edges, 1, np.pi / 180, 400, None, 0, 0)

    filtered_lines = []
    # draw the lines
    if lines is not None:
        for line in lines:
            theta = line[0][1]
            if not math.isclose(theta, math.pi / 2, rel_tol=EPSILON):
                filtered_lines.append(line)

    if len(filtered_lines) > 0: 
        for line_1 in filtered_lines:
            theta_1 = line_1[0][1]
            for line_2 in filtered_lines:
                theta_2 = line_2[0][1]
                
                angles_opposite = math.isclose(abs(math.pi - theta_1), theta_2, rel_tol=EPSILON)

                if angles_opposite:
                    poi_x = point_of_intersection(line_1, line_2)[0]
                    cols = image.shape[1]

                    if math.isclose(poi_x, cols, rel_tol=MIDDLE_TOLERANCE):
                        return (line_1, line_2)


def main():
    screen = cv.imread('./test.png')
    guitar_lines = find_guitar(screen) 
    print(guitar_lines)

    cv.imshow('wimdow', screen)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()