import pdb
import time
import cv2 as cv
import numpy as np

from frame import grab_image, grab_frame_from_video
from threading import Event

SCALE_FROM = (2560, 1440)
SCALE_TO = (1920, 1080)


def scale(points, scale_from, scale_to):
    for idx, point in enumerate(points):
        points[idx] = (round(point[0] * scale_to[0] / scale_from[0]), round(point[1] * scale_to[1] / scale_from[1]))
    return points


# first is top-left, working clockwise
BOUNDING_POLYGON = scale(np.array([
    (963, 920),
    (1596, 920),
    (1650, 1020),
    (910, 1020),
]), SCALE_FROM, SCALE_TO)


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


class Note:
    def __init__(self):
        self.positions = []
        self.velocities = []
        self.update_time = None


def match_notes(tracked_notes, unmatched_notes, frame_time):
    unmatched_notes.sort(key=lambda x: x[1], reverse=True)
    notes_to_delete = set()
    for t_idx, t_note in enumerate(tracked_notes):
        t_y = t_note.positions[-1][1]
        found = False
        for u_idx, u_note in enumerate(unmatched_notes):
            u_y = u_note[1]
            if t_y <= u_y:
                u_center = (u_note[0] + u_note[2], u_note[1] + u_note[3])
                t_pos = t_note.positions[-1]
                t_center = (t_pos[0] + t_pos[2], t_pos[1] + t_pos[3])

                delta_position = (u_center[0] - t_center[0], u_center[1] - t_center[1])
                delta_time = frame_time - t_note.update_time
                velocity = (delta_position[0] / delta_time, delta_position[1] / delta_time)

                t_note.positions.append(u_note)
                t_note.velocities.append(velocity)
                t_note.update_time = frame_time

                del unmatched_notes[u_idx]
                found = True
                break

        if not found:
            notes_to_delete.add(t_idx)

    tracked_notes = [t_note for t_idx, t_note in enumerate(tracked_notes) if t_idx not in notes_to_delete]

    for u_note in reversed(unmatched_notes):
        note = Note()
        note.positions.append(u_note)
        note.update_time = frame_time
        tracked_notes.insert(0, note)

    return tracked_notes


def main():
    stop_event = Event()
    # frames = grab_image("./testing-files/test_image.png")
    frames = grab_frame_from_video(stop_event, "./testing-files/some_might_say_second_vid.mkv")
    for f in frames:
        frame_time = time.monotonic_ns()
        frame_hsv = cv.cvtColor(f, cv.COLOR_BGR2HSV)
        for idx, (bounding_polygon, note_mask, area_threshold) in enumerate(zip(BOUNDING_POLYGONS, NOTE_MASKS, NOTE_AREA_THRESHOLDS)):
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
            area_threshold = round(area_threshold * (SCALE_TO[0] * SCALE_TO[1]) / (SCALE_FROM[0] * SCALE_FROM[1]))
            filtered_bounding_boxes = [bb for bb in bounding_boxes if bb[2] * bb[3] > area_threshold]

            converted_bounding_boxes = [np.array(((bb[0], bb[1]), (bb[0] + bb[2], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), (bb[0], bb[1] + bb[3]))) for bb in filtered_bounding_boxes]

            cv.drawContours(notes, converted_bounding_boxes, -1, (255, 0, 0), -1, cv.LINE_AA)

            note_tracking = NOTE_TRACKING[idx]
            note_tracking = match_notes(note_tracking, filtered_bounding_boxes, frame_time)
            NOTE_TRACKING[idx] = note_tracking

            cv.imshow("cropped", notes)

            key = cv.waitKey(0)
            while key != ord("q"):
                key = cv.waitKey(0)


if __name__ == "__main__":
    main()
