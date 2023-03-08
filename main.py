import math
import cv2 as cv
import numpy as np

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
            if not math.isclose(theta, math.pi / 2, rel_tol=1e-5):
                filtered_lines.append(line)

    guitar = []
    if len(filtered_lines) > 1:
        # two sum to pi / 2
        target = float(math.pi / 2)
        angles = {}
        for index, line in enumerate(filtered_lines):
            theta = line[0][1]
            if theta in angles.keys():
                guitar.append(filtered_lines[index])
                guitar.append(filtered_lines[angles[theta]])
            else:
                angles[target - theta] = index

    print(filtered_lines)

    return drawn_outline 

def main():
    screen = cv.imread('./test.png')
    cdst = find_guitar(screen) 

    cv.imshow('wimdow', cdst)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()