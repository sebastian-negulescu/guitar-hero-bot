#!/usr/bin/env python3
import itertools
from pydbus import SessionBus
from gi.repository import GLib
import cv2

#
# ---------- Portal setup ----------
#

loop = GLib.MainLoop()
bus = SessionBus()

portal = bus.get(
    "org.freedesktop.portal.Desktop",
    "/org/freedesktop/portal/desktop",
)
sc = portal["org.freedesktop.portal.ScreenCast"]

counter = itertools.count(1)
def new_token(prefix: str) -> str:
    # Valid DBus path element: [A-Za-z_][A-Za-z0-9_]*
    return f"{prefix}_{next(counter)}"

create_token = new_token("pycap")
select_token = new_token("pycap")
start_token  = new_token("pycap")

state = {
    "session_handle": None,
    "create_request": None,
    "select_request": None,
    "start_request": None,
    "node_id": None,
}

def log(*a):
    print(*a, flush=True)

#
# ---------- Handle portal responses ----------
#

def on_request_response(sender, object_path, iface, signal, params):
    if iface != "org.freedesktop.portal.Request" or signal != "Response":
        return

    response, results = params
    log(f"\n== Response {response} from {object_path} ==")
    log("   Results:", results)

    # --- CreateSession ---
    if object_path == state["create_request"] or object_path.endswith(create_token):
        if response != 0:
            log("CreateSession failed:", results)
            loop.quit()
            return

        session_handle = results.get("session_handle")
        if not session_handle:
            log("No session_handle in CreateSession results!")
            loop.quit()
            return

        state["session_handle"] = session_handle
        log("Session handle:", session_handle)

        # SelectSources(session_handle, options)
        opts = {
            "handle_token": GLib.Variant("s", select_token),
            "types": GLib.Variant("u", 1),        # 1 = MONITOR (2=WINDOW, 4=VIRTUAL)
            "multiple": GLib.Variant("b", False),
            "cursor_mode": GLib.Variant("u", 1),  # hide cursor (safer default)
        }
        log("\n→ Calling SelectSources")
        state["select_request"] = sc.SelectSources(session_handle, opts)
        log("SelectSources request:", state["select_request"])
        return

    # --- SelectSources ---
    if object_path == state["select_request"] or object_path.endswith(select_token):
        if response != 0:
            log("SelectSources failed:", results)
            loop.quit()
            return

        # Start(session_handle, parent_window, options)
        opts = {
            "handle_token": GLib.Variant("s", start_token),
        }
        log("\n→ Calling Start")
        state["start_request"] = sc.Start(state["session_handle"], "s", opts)
        log("Start request:", state["start_request"])
        return

    # --- Start ---
    if object_path == state["start_request"] or object_path.endswith(start_token):
        if response != 0:
            log("Start failed:", results)
            loop.quit()
            return

        streams = results.get("streams", [])
        if not streams:
            log("No streams in Start results")
            loop.quit()
            return

        first = streams[0]
        node_id = first[0]
        state["node_id"] = int(node_id)
        log("Got PipeWire node id:", state["node_id"])

        # We now have everything we need; leave the GLib loop
        loop.quit()
        return

# Subscribe to Request::Response
bus.subscribe(
    sender="org.freedesktop.portal.Desktop",
    iface="org.freedesktop.portal.Request",
    signal="Response",
    signal_fired=on_request_response,
)

#
# ---------- Start screencast session ----------
#

create_opts = {
    "handle_token": GLib.Variant("s", create_token),
    "session_handle_token": GLib.Variant("s", new_token("session")),
}

log("→ Calling CreateSession")
state["create_request"] = sc.CreateSession(create_opts)
log("CreateSession request:", state["create_request"])
log("Waiting for portal dialog… (pick a screen/window, hit Share)")

try:
    loop.run()
except KeyboardInterrupt:
    log("Interrupted while setting up screencast")
    exit(1)

if state["node_id"] is None or state["session_handle"] is None:
    log("Failed to obtain PipeWire node id or session handle")
    exit(1)

#
# ---------- OpenCV + GStreamer from PipeWire ----------
#

node_id = state["node_id"]
pipeline = (
    f"pipewiresrc path={node_id} ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink drop=1")

log("\n→ Opening OpenCV capture with pipeline:")
log("   ", pipeline)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 60.0, (2560,  1440))

if not cap.isOpened():
    log("Failed to open GStreamer pipeline via OpenCV")
    # Clean up the session before exiting
    session_obj = bus.get("org.freedesktop.portal.Desktop", state["session_handle"])
    session_obj.Close()
    exit(1)

log("Capture running. Press 'q' to quit.")

frames = []
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 0)
        # write the flipped frame
        out.write(frame)

        # cv2.imshow("Hyprland Screencast", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

    # Properly close the portal session to stop screencast
    log("\n→ Closing portal session")
    session_obj = bus.get("org.freedesktop.portal.Desktop", state["session_handle"])
    session_obj.Close()
    log("Done.")

