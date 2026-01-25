#!/usr/bin/env python3
import itertools
from pydbus import SessionBus
from gi.repository import GLib
import cv2

loop = GLib.MainLoop()
bus = SessionBus()

portal = bus.get(
    "org.freedesktop.portal.Desktop",
    "/org/freedesktop/portal/desktop",
)
sc = portal["org.freedesktop.portal.ScreenCast"]

counter = itertools.count(1)
def new_token(prefix: str) -> str:
    return f"{prefix}_{next(counter)}"  # valid as a DBus path element

create_token = new_token("pycap")
select_token = new_token("pycap")
start_token  = new_token("pycap")

state = {
    "session_handle": None,
    "create_request": None,
    "select_request": None,
    "start_request": None,
    "node_id": None,
    "pw_fd": None,
}

def log(*a):
    print(*a, flush=True)

def on_request_response(sender, object_path, iface, signal, params):
    if iface != "org.freedesktop.portal.Request" or signal != "Response":
        return

    response, results = params
    log(f"\n== Response {response} from {object_path} ==")
    log("   Results:", results)

    # CREATE SESSION
    if object_path == state["create_request"] or object_path.endswith(create_token):
        if response != 0:
            log("CreateSession failed:", results)
            loop.quit()
            return

        session_handle = results.get("session_handle")
        if not session_handle:
            log("No session_handle in CreateSession results")
            loop.quit()
            return

        state["session_handle"] = session_handle
        log("Session handle:", session_handle)

        opts = {
            "handle_token": GLib.Variant("s", select_token),
            "types": GLib.Variant("u", 1),        # 1 = monitor
            "multiple": GLib.Variant("b", False),
            "cursor_mode": GLib.Variant("u", 1),  # hide cursor
        }
        log("\n→ Calling SelectSources")
        state["select_request"] = sc.SelectSources(session_handle, opts)
        log("SelectSources request:", state["select_request"])
        return

    # SELECT SOURCES
    if object_path == state["select_request"] or object_path.endswith(select_token):
        if response != 0:
            log("SelectSources failed:", results)
            loop.quit()
            return

        opts = {
            "handle_token": GLib.Variant("s", start_token),
        }
        log("\n→ Calling Start")
        state["start_request"] = sc.Start(state["session_handle"], "", opts)
        log("Start request:", state["start_request"])
        return

    # START
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

        # streams is a list of (node_id, props) tuples
        node_id, props = streams[0]
        state["node_id"] = int(node_id)
        log("\n🎉 Got PipeWire node id:", state["node_id"])
        log("   Stream props:", props)

        # Get fd for private PipeWire remote
        pw_fd = sc.OpenPipeWireRemote(state["session_handle"], {})
        state["pw_fd"] = int(pw_fd)
        log("Got PipeWire remote fd:", state["pw_fd"])

        # Done with portal setup; exit GLib loop
        loop.quit()
        return


# Subscribe to Request::Response
bus.subscribe(
    sender="org.freedesktop.portal.Desktop",
    iface="org.freedesktop.portal.Request",
    signal="Response",
    signal_fired=on_request_response,
)

# Start portal session
create_opts = {
    "handle_token": GLib.Variant("s", create_token),
    "session_handle_token": GLib.Variant("s", new_token("session")),
}

log("→ Calling CreateSession")
state["create_request"] = sc.CreateSession(create_opts)
log("CreateSession request:", state["create_request"])
log("Waiting for portal dialog… (select monitor/window and Share)")

try:
    loop.run()   # wait until we have node_id
except KeyboardInterrupt:
    log("Interrupted while setting up screencast")
    exit(1)

if state["node_id"] is None or state["session_handle"] is None:
    log("Failed to obtain node_id or session_handle")
    exit(1)

# ---- Now use OpenCV in the SAME process ----

node_id = state["node_id"]
pw_fd   = state["pw_fd"]

pipeline = (
    f"pipewiresrc fd={pw_fd} path={node_id} ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink drop=1"
)

log("\n→ Opening OpenCV capture:")
log("   ", pipeline)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    log("Failed to open GStreamer pipeline via OpenCV")
    # Close the portal session before exiting
    session_obj = bus.get("org.freedesktop.portal.Desktop", state["session_handle"])
    session_obj.Close()
    exit(1)

log("Capture running. Press 'q' or Esc to quit.")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            # brief glitches can happen; just continue
            continue

        cv2.imshow("Hyprland Screencast", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()

    log("\n→ Closing portal session")
    session_obj = bus.get("org.freedesktop.portal.Desktop", state["session_handle"])
    session_obj.Close()
    log("Done.")

