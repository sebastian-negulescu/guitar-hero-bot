import pdb
import time
import cv2 as cv
import numpy as np

from frame import grab_image, grab_frame_from_video
from threading import Event

SCALE_FROM = (2560, 1440)
# SCALE_TO = (2560, 1440)
SCALE_TO = (1920, 1080)


def scale(points, scale_from, scale_to):
    scaled_points = []
    for point in points:
        scaled_point = []
        for dim, p in enumerate(point):
            scaled_point.append(round(p * scale_to[dim] / scale_from[dim]))
        scaled_points.append(tuple(scaled_point))
    return scaled_points


# first is top-left, working clockwise
GUITAR_SECTION = scale(np.array([
    (963, 920),
    (1596, 920),
    (1650, 1020),
    (910, 1020),
]), SCALE_FROM, SCALE_TO)


def create_strings(guitar):
    num_strings = 5
    spacing = np.array(
        (guitar[1][0] - guitar[0][0],
         guitar[2][0] - guitar[3][0])
    ) // num_strings

    strings = []
    last_edge = np.array((guitar[0], guitar[3]))
    for i in range(num_strings):
        strings.append(
            np.array([last_edge[0], last_edge[0] + np.array((spacing[0], 0)),
                      last_edge[1] + np.array((spacing[1], 0)), last_edge[1]]))
        last_edge = np.array((strings[-1][1],
                              strings[-1][2]))

    return strings


STRING_SECTIONS = create_strings(GUITAR_SECTION)

NOTE_MASKS = [
    (((115, 135),), 60, 40),
    (((350, 360), (0, 10)), 60, 40),
    (((50, 65),), 60, 40),
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


def create_note_threshold(bottom_y):
    RELATIVE_BOUNDS = scale([(29, 142)], SCALE_FROM, SCALE_TO)[0]
    w = scale([[0.214 * bottom_y + 126.6]], SCALE_FROM, SCALE_TO)[0][0]
    h = RELATIVE_BOUNDS[0] * w / RELATIVE_BOUNDS[1]
    return w * h * 0.5


note_tracking = [
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


def draw_bb_on_note(notes, filtered_note_bb):
    converted_bounding_boxes = [np.array(((bb[0], bb[1]),
                                          (bb[0] + bb[2], bb[1]),
                                          (bb[0] + bb[2], bb[1] + bb[3]),
                                          (bb[0], bb[1] + bb[3])))
                                for bb in filtered_note_bb]
    cv.drawContours(notes,
                    converted_bounding_boxes,
                    -1,
                    (255, 0, 0),
                    -1,
                    cv.LINE_AA)

    return notes


def main():
    stop_event = Event()
    # frames = grab_image("./testing-files/red_yellow_red.png")
    frames = grab_frame_from_video(stop_event, "./testing-files/some_might_say.mkv")
    for f in frames:
        frame_time = time.monotonic_ns()
        frame_hsv = cv.cvtColor(f, cv.COLOR_BGR2HSV)
        for idx, (string, note_mask) in enumerate(zip(STRING_SECTIONS, NOTE_MASKS)):
            # get cropped note lane
            string_bb = cv.boundingRect(string)
            x, y, w, h = string_bb
            string_cropped = frame_hsv[y:y+h, x:x+w].copy()

            relative_string = string - string.min(axis=0)
            mask = np.zeros(string_cropped.shape[:2], np.uint8)
            cv.drawContours(
                mask, [relative_string], -1, (255, 255, 255), -1, cv.LINE_AA)

            string_masked = cv.bitwise_and(
                string_cropped, string_cropped, mask=mask)

            # apply note colour mask
            s_min = round(note_mask[1] * 255 / 100)
            v_min = round(note_mask[2] * 255 / 100)
            note_masked = np.zeros(string_masked.shape, np.uint8)
            for h in note_mask[0]:
                h_min = h[0] // 2
                h_max = h[1] // 2

                h_mask = cv.inRange(string_masked,
                                    np.array((h_min, s_min, v_min)),
                                    np.array((h_max, 255, 255)))
                part_note_masked = cv.bitwise_and(
                    string_masked, string_masked, mask=h_mask)
                note_masked = cv.bitwise_or(part_note_masked, note_masked)

            # get area of solid regions
            notes = cv.cvtColor(note_masked, cv.COLOR_HSV2BGR)
            notes_grey = cv.cvtColor(notes, cv.COLOR_BGR2GRAY)
            _, notes_grey = cv.threshold(notes_grey, 10, 255, cv.THRESH_BINARY)

            note_contours, _ = cv.findContours(
                notes_grey, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            note_bb = [cv.boundingRect(contour) for contour in note_contours]
            filtered_note_bb = [bb for bb in note_bb
                                if bb[2] * bb[3] > create_note_threshold(bb[1] + bb[3])]

            # show the matched areas on the image
            note_contours_scaled = []
            for contour in note_contours:
                scaled_contour = np.array([[point[0][0] + x, point[0][1] + y] for point in contour])
                note_contours_scaled.append(scaled_contour)
            cv.drawContours(f, note_contours_scaled, -1, (255, 0, 0), -1, cv.LINE_AA)

            note_tracking[idx] = match_notes(note_tracking[idx], filtered_note_bb, frame_time)

        cv.imshow("frame", f)

        key = cv.waitKey(0)
        while key != ord("q"):
            key = cv.waitKey(0)


if __name__ == "__main__":
    main()
