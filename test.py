#!/usr/bin/env python3

import itertools
from pydbus import SessionBus
from gi.repository import GLib

loop = GLib.MainLoop()
bus = SessionBus()

# Portal root object
portal = bus.get(
    "org.freedesktop.portal.Desktop",
    "/org/freedesktop/portal/desktop",
)

# ScreenCast interface proxy
sc = portal["org.freedesktop.portal.ScreenCast"]

# Simple, valid tokens: [A-Za-z_][A-Za-z0-9_]*, no dashes
counter = itertools.count(1)


def new_token(prefix: str) -> str:
    return f"{prefix}_{next(counter)}"


create_token = new_token("pydbus")
select_token = new_token("pydbus")
start_token = new_token("pydbus")

state = {
    "session_handle": None,        # /org/freedesktop/portal/desktop/session/...
    "create_request": None,        # request handle from CreateSession
    "select_request": None,        # request handle from SelectSources
    "start_request":  None,        # request handle from Start
}


def log(*a):
    print(*a, flush=True)


def on_request_response(sender, object_path, iface, signal, params):
    """
    sender: unique bus name (":1.42")
    object_path: request path, e.g. /org/freedesktop/portal/desktop/request/pydbus_1
    iface: "org.freedesktop.portal.Request"
    signal: "Response"
    params: (response: uint32, results: dict)
    """
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
            log("No session_handle in CreateSession results!")
            loop.quit()
            return

        state["session_handle"] = session_handle
        log("Session handle:", session_handle)

        # Now call SelectSources(session_handle, options)
        opts = {
            "handle_token": GLib.Variant("s", select_token),
            "types": GLib.Variant("u", 1),          # 1 = MONITOR; (2 = WINDOW, 4 = VIRTUAL)
            "multiple": GLib.Variant("b", False),
            "cursor_mode": GLib.Variant("u", 1),    # 1 = Hidden (always safe)
        }

        log("\n→ Calling SelectSources")
        state["select_request"] = sc.SelectSources(session_handle, opts)
        log("SelectSources request handle:", state["select_request"])
        return

    # SELECT SOURCES
    if object_path == state["select_request"] or object_path.endswith(select_token):
        if response != 0:
            log("SelectSources failed:", results)
            loop.quit()
            return

        # Now Start(session_handle, parent_window, options)
        opts = {
            "handle_token": GLib.Variant("s", start_token),
        }

        log("\n→ Calling Start")
        state["start_request"] = sc.Start(state["session_handle"], "", opts)
        log("Start request handle:", state["start_request"])
        return

    # START
    if object_path == state["start_request"] or object_path.endswith(start_token):
        print("Start results:", results)  # <-- add this temporarily
        if response != 0:
            log("Start failed:", results)
            loop.quit()
            return

        streams = results.get("streams", [])
        if not streams:
            log("No streams returned in Start results")
            loop.quit()
            return

        first = streams[0]
        log("\nPipeWire node id:", first)
        # stop_screencast()
        input("pausing...")

        # You can now connect to this node with PipeWire
        loop.quit()


# Stop session function
def stop_screencast():
    session = bus.get(
        "org.freedesktop.portal.Desktop",
        state["session_handle"]
    )
    session.Close()


# Subscribe to all Request::Response signals from the portal
bus.subscribe(
    sender="org.freedesktop.portal.Desktop",
    iface="org.freedesktop.portal.Request",
    signal="Response",
    signal_fired=on_request_response,
)

# 1) CreateSession(options) → returns request handle
create_opts = {
    "handle_token": GLib.Variant("s", create_token),
    "session_handle_token": GLib.Variant("s", new_token("session")),
}

log("→ Calling CreateSession")
state["create_request"] = sc.CreateSession(create_opts)
log("CreateSession request handle:", state["create_request"])

log("Waiting for portal responses… (you should see a share dialog)")
try:
    loop.run()
except KeyboardInterrupt:
    log("Interrupted")
