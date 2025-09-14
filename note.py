import cv2 as cv
import numpy as np

from frame import grab_image

SCALE_FROM = (2160, 1440)
SCALE_TO = (1920, 1080)

# first is top-left, working clockwise
BOUNDING_POLYGON = np.array([
    (1080, 710),
    (1480, 710),
    (1650, 1020),
    (910, 1020),
])


def create_bounding_polygons(bounding_polygon):
    num_polygons = 5
    spacing = np.array((bounding_polygon[1][0] - bounding_polygon[0][0], bounding_polygon[2][0] - bounding_polygon[3][0])) // num_polygons
    bounding_polygons = []
    last_edge = np.array((bounding_polygon[0], bounding_polygon[3]))
    for i in range(num_polygons):
        bounding_polygons.append(np.array([last_edge[0], last_edge[0] + np.array((spacing[0], 0)), last_edge[1] + np.array((spacing[1], 0)), last_edge[1]]))
        last_edge = np.array((bounding_polygons[-1][1], bounding_polygons[-1][2]))

    return bounding_polygons


BOUNDING_POLYGONS = create_bounding_polygons(BOUNDING_POLYGON)

NOTE_MASKS = [
    ((115, 135), 60, 40),
    ((350, 10), 60, 40),
    ((50, 65), 70, 40),
    ((0, 0), 0, 0),
    ((0, 0), 0, 0),
]


def main():
    frames = grab_image("./testing-files/2025-09-13-150719_hyprshot.png")
    for f in frames:
        frame_hsv = cv.cvtColor(f, cv.COLOR_BGR2HSV)
        for bounding_polygon, note_mask in zip(BOUNDING_POLYGONS, NOTE_MASKS):
            # get cropped note lane
            bounding_box = cv.boundingRect(bounding_polygon)
            x, y, w, h = bounding_box
            cropped = frame_hsv[y:y+h, x:x+w].copy()

            relative_polygon = bounding_polygon - bounding_polygon.min(axis=0)
            mask = np.zeros(cropped.shape[:2], np.uint8)
            cv.drawContours(mask, [relative_polygon], -1, (255, 255, 255), -1, cv.LINE_AA)

            bound_masked = cv.bitwise_and(cropped, cropped, mask=mask)

            # apply other mask
            h_min = note_mask[0][0] // 2
            h_max = note_mask[0][1] // 2
            s_min = round(note_mask[1] * 255 / 100)
            v_min = round(note_mask[2] * 255 / 100)

            min_mask = cv.inRange(bound_masked, np.array((h_min, s_min, v_min)), np.array((179, 255, 255)))
            min_masked = cv.bitwise_and(bound_masked, bound_masked, mask=min_mask)

            max_mask = cv.inRange(bound_masked, np.array((0, s_min, v_min)), np.array((h_max, 255, 255)))
            max_masked = cv.bitwise_and(bound_masked, bound_masked, mask=max_mask)

            note_masked = cv.bitwise_or(min_masked, max_masked)

            # get area of solid regions

            cv.imshow("frame", note_masked)

            key = cv.waitKey(0)
            while key != ord("q"):
                key = cv.waitKey(0)


if __name__ == "__main__":
    main()
