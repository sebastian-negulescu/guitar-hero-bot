import math
import cv2 as cv
import numpy as np

def find_guitar( image ):
    edges = cv.Canny( image, 50, 200, None, 3 )

    # Copy edges to the images that will display the results in BGR
    drawn_outline = cv.cvtColor( edges, cv.COLOR_GRAY2BGR )

    lines = cv.HoughLines( edges, 1, np.pi / 180, 400, None, 0, 0 )

    # draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 5000*(-b)), int(y0 + 5000*(a)))
            pt2 = (int(x0 - 5000*(-b)), int(y0 - 5000*(a)))
            cv.line(drawn_outline, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

    return drawn_outline 

def main():
    screen = cv.imread('./test.png')
    cdst = find_guitar(screen) 

    cv.imshow('wimdow', cdst)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()