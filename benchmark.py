import time
import capture

c = capture.ScreenCapture().generate_capture_object()
last = time.time()
while True:
    _, _ = c.read()
    now = time.time()
    print(now - last)
    last = now
