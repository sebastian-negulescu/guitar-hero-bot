import pdb
import time
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
    (((115, 135),), 60, 40),
    (((350, 360), (0, 10)), 60, 40),
    (((50, 65),), 70, 40),
    (((360, 360),), 100, 100),
    (((360, 360),), 100, 100),
]

NOTE_AREA_THRESHOLDS = [
    256,
    256,
    256,
    256,
    256,
]


NOTE_TRACKING = [
    [],
    [],
    [],
    [],
    [],
]


def main():
    frames = grab_image("./testing-files/2025-09-13-150719_hyprshot.png")
    for f in frames:
        frame_hsv = cv.cvtColor(f, cv.COLOR_BGR2HSV)
        for bounding_polygon, note_mask, area_threshold, note_tracking in zip(BOUNDING_POLYGONS, NOTE_MASKS, NOTE_AREA_THRESHOLDS, NOTE_TRACKING):
            # get cropped note lane
            bounding_box = cv.boundingRect(bounding_polygon)
            x, y, w, h = bounding_box
            cropped = frame_hsv[y:y+h, x:x+w].copy()

            relative_polygon = bounding_polygon - bounding_polygon.min(axis=0)
            mask = np.zeros(cropped.shape[:2], np.uint8)
            cv.drawContours(mask, [relative_polygon], -1, (255, 255, 255), -1, cv.LINE_AA)

            bound_masked = cv.bitwise_and(cropped, cropped, mask=mask)

            # apply other mask
            s_min = round(note_mask[1] * 255 / 100)
            v_min = round(note_mask[2] * 255 / 100)
            note_masked = np.zeros(bound_masked.shape, np.uint8)
            for h in note_mask[0]:
                h_min = h[0] // 2
                h_max = h[1] // 2

                h_mask = cv.inRange(bound_masked, np.array((h_min, s_min, v_min)), np.array((h_max, 255, 255)))
                part_note_masked = cv.bitwise_and(bound_masked, bound_masked, mask=h_mask)
                note_masked = cv.bitwise_or(part_note_masked, note_masked)

            # get area of solid regions
            notes = cv.cvtColor(note_masked, cv.COLOR_HSV2BGR)
            notes_grey = cv.cvtColor(notes, cv.COLOR_BGR2GRAY)
            _, notes_grey = cv.threshold(notes_grey, 10, 255, cv.THRESH_BINARY)

            contours, hierarchy = cv.findContours(notes_grey, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            bounding_boxes = [cv.boundingRect(contour) for contour in contours]
            filtered_bounding_boxes = []
            for idx, bb in enumerate(bounding_boxes):
                x, y, w, h = bb
                area = w * h
                if area >= area_threshold:
                    filtered_bounding_boxes.append(
                        np.array(((x, y),
                                  (x + w, y),
                                  (x + w, y + h),
                                  (x, y + h))))

            unmatched_bounding_boxes = sorted(filtered_bounding_boxes, key=lambda x: x[0][1], reverse=True)
            tracked_notes_to_delete = set()
            for t_idx, tracked_note in enumerate(note_tracking):
                t_pos = tracked_note[0][1]
                found = False
                for n_idx, bb in unmatched_bounding_boxes:
                    n_pos = bb[0][1]
                    if t_pos < n_pos:
                        # new note position is farther down screen than tracked position
                        note_tracking[t_idx] = bb
                        del unmatched_bounding_boxes[n_idx]
                        found = True
                        break
                if not found:
                    tracked_notes_to_delete.add(t_idx)

            for idx in tracked_notes_to_delete:
                del note_tracking[idx]

            for note in reversed(unmatched_bounding_boxes):
                note_tracking.insert(0, note)

            cv.drawContours(notes, note_tracking, -1, (0, 255, 0), -1, cv.LINE_AA)

            cv.imshow("frame", notes)

            key = cv.waitKey(0)
            while key != ord("q"):
                key = cv.waitKey(0)


if __name__ == "__main__":
    main()
