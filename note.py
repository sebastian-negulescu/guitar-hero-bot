import pdb
import time
import math
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

BASELINE = scale([[1260]], SCALE_FROM[1:], SCALE_TO[1:])[0][0]


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


def check_note_proportions(note_bb):
    bottom_y = note_bb[1] + note_bb[3]

    RELATIVE_BOUNDS = scale([(29, 142)], SCALE_FROM, SCALE_TO)[0]
    w = scale([[0.214 * bottom_y + 126.6]], SCALE_FROM, SCALE_TO)[0][0]
    h = RELATIVE_BOUNDS[0] * w / RELATIVE_BOUNDS[1]

    return (math.isclose(w, note_bb[2], rel_tol=0.2) and
            math.isclose(h, note_bb[3], rel_tol=0.2))


def match_notes(tracked_notes, unmatched_notes, frame_time):
    unmatched_notes.sort(key=lambda note: note[1])
    notes_to_delete = set()

    for t_idx, t_note in enumerate(tracked_notes):
        t_y = t_note.positions[-1][1]
        found = False
        for u_idx, u_note in enumerate(unmatched_notes):
            u_y = u_note[1]
            if t_y <= u_y:
                u_pos = (u_note[0] + u_note[2] / 2, u_y)
                t_note.add_position(u_pos, frame_time)
                del unmatched_notes[u_idx]
                found = True
                break

        if not found:
            notes_to_delete.add(t_idx)

    predicted_times = []
    for t_idx in notes_to_delete:
        last_pos = tracked_notes[t_idx].positions[-1]
        last_time = tracked_notes[t_idx].times[-1]
        time_delta = last_time - tracked_notes[t_idx].times[0]
        pos_delta = last_pos[1] - tracked_notes[t_idx].positions[0][1]
        velocity = pos_delta / time_delta
        delta_pos = BASELINE - last_pos[1]
        delta_time = delta_pos / velocity
        predicted_times.append(last_time + delta_time)
        print("TIMES", last_time, tracked_notes[t_idx].times[0])
        print("PRED TIME", last_time + delta_time)
        print("TIME DELTA", time_delta)
        print("POS DELTA", last_pos[1] - tracked_notes[t_idx].positions[0][1])
        print("VELOCITY", pos_delta / time_delta)
        print("POS", last_pos)

    tracked_notes = [t_note for t_idx, t_note in enumerate(tracked_notes)
                     if t_idx not in notes_to_delete]

    for u_note in reversed(unmatched_notes):
        u_pos = (u_note[0] + u_note[2] / 2, u_note[1])
        note = Note(u_pos, frame_time)
        tracked_notes.insert(0, note)

    return tracked_notes, predicted_times


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


class Note:
    def __init__(self, pos, t):
        self.positions = [pos]
        self.times = [t]
        self.velocities = []

    def add_position(self, pos, t):
        self.positions.append(pos)
        self.times.append(t)


def main():
    note_tracking = [[]] * 5
    time_tracking = [[]] * 5

    stop_event = Event()
    # frames = grab_image("./testing-files/red_yellow_red.png")
    frames = grab_frame_from_video(stop_event, "./testing-files/some_might_say.mkv")
    for frame_count, f in enumerate(frames):
        frame_time = time.monotonic_ns()
        frame_hsv = cv.cvtColor(f, cv.COLOR_BGR2HSV)
        for note_idx, (string, note_mask) in enumerate(zip(STRING_SECTIONS, NOTE_MASKS)):
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
            note_bbs = [cv.boundingRect(contour) for contour in note_contours]
            filtered_note_bbs = list(filter(check_note_proportions, note_bbs))

            # show the matched areas on the image
            note_contours_scaled = []
            for contour in note_contours:
                scaled_contour = np.array([[point[0][0] + x, point[0][1] + y] for point in contour])
                note_contours_scaled.append(scaled_contour)
            cv.drawContours(f, note_contours_scaled, -1, (255, 0, 0), -1, cv.LINE_AA)

            # draw the bounding box on the image
            scaled_filtered_note_bbs = []
            for bb in filtered_note_bbs:
                scaled_bb = (bb[0] + x, bb[1] + y, bb[2], bb[3])
                cv.rectangle(f,
                             (scaled_bb[0], scaled_bb[1]),
                             (scaled_bb[0] + scaled_bb[2],
                              scaled_bb[1] + scaled_bb[3]),
                             (0, 255, 0),
                             2,
                             cv.LINE_AA)
                scaled_filtered_note_bbs.append(scaled_bb)

            note_tracking[note_idx], predicted_times = match_notes(
                note_tracking[note_idx], scaled_filtered_note_bbs, frame_count)
            time_tracking[note_idx].extend(predicted_times)

        for note, note_time_tracking in enumerate(time_tracking):
            for idx in reversed(range(len(note_time_tracking))):
                estimate = note_time_tracking[idx]
                if frame_count >= estimate:
                    print("HIT")
                    del note_time_tracking[idx]
                    break
        # print(sum(map(lambda x: len(x), note_tracking)))
        cv.imshow("frame", f)

        key = cv.waitKey(0)
        while key != ord("q"):
            key = cv.waitKey(0)


if __name__ == "__main__":
    main()
