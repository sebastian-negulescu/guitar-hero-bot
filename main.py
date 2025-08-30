import time
import cv2 as cv

from multiprocessing import Process, Event

from streaming_server import run_screen_record

NS_TO_S = 1 / 1_000_000_000
target_ns = 17_000_000

if __name__ == "__main__":

    stop_event = Event()
    start_event = Event()
    footage_process = Process(target=run_screen_record,
                              args=(start_event, stop_event,))
    footage_process.start()
    start_event.wait()

    start_time = time.time()

    while time.time() - start_time < 60:
        try:
            cap = cv.VideoCapture("stream.flv")
            num_frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
            cap.set(cv.CAP_PROP_POS_FRAMES, num_frames - 1)
            timer = time.monotonic_ns()
            while cap.isOpened() and time.time() - start_time < 60:
                ret, frame = cap.read()
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                cv.imshow('frame', gray)
                if cv.waitKey(1) == ord('q'):
                    break

                time.sleep(((time.monotonic_ns() - timer) % target_ns) * NS_TO_S)

        except Exception:
            pass

    stop_event.set()
    footage_process.join()
