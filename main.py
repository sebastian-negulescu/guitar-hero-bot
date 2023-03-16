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

    guitar = None
    if len(filtered_lines) > 0: 
        for line_1 in filtered_lines:
            theta_1 = line_1[0][1]
            for line_2 in filtered_lines:
                theta_2 = line_2[0][1]
                if math.isclose(math.pi - theta_1, theta_2, rel_tol=1e-3):
                    guitar = (line_1, line_2)
                    break

    print(filtered_lines)
    print(guitar)

    return drawn_outline, guitar

def main():
    screen = cv.imread('./test.png')
    cdst, _ = find_guitar(screen) 

    cv.imshow('wimdow', cdst)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()