from pydbus import SessionBus
from gi.repository import GLib
import cv2 as cv
import itertools

class ScreenCapture:
    DESKTOP_PATH = "/org/freedesktop/portal/desktop"
    PORTAL_PREFIX = "org.freedesktop.portal"
    DESKTOP_SUFFIX = "Desktop"
    REQUEST_SUFFIX = "Request"
    SCREENCAST_SUFFIX = "ScreenCast"

    def __init__(self):
        self.__loop = GLib.MainLoop()
        self.__bus = SessionBus()

        self.__portal = self.__bus.get(
            "org.freedesktop.portal.Desktop",
            "/org/freedesktop/portal/desktop",
        )
        self.__screencast = self.__portal["org.freedesktop.portal.ScreenCast"]

        self.__state = {}
        self.__handler = None

        self.__bus.subscribe(
            sender=f"{self.PORTAL_PREFIX}.{self.DESKTOP_SUFFIX}",
            iface=f"{self.PORTAL_PREFIX}.{self.REQUEST_SUFFIX}",
            signal="Response",
            signal_fired=self.__handle_response,
        )

    def __generate_token(self):
        id = itertools.count(1)
        return f"screencapture_{next(id)}"

    def __handle_create_session(self, object_path, response, results):
        if object_path != self.__state["create_request"]:
            return self.__handle_create_session

        if response != 0:
            return None

        session_handle = results.get("session_handle")
        if not session_handle:
            return None

        self.__state["session_handle"] = session_handle

        opts = {
            "handle_token": GLib.Variant("s", self.__generate_token()),
            "types": GLib.Variant("u", 1),
            "multiple": GLib.Variant("b", False),
            "cursor_mode": GLib.Variant("u", 1),
        }
        self.__state["select_request"] = self.__screencast.SelectSources(session_handle, opts)
        return self.__handle_select_sources

    def __handle_select_sources(self, object_path, response, results):
        if object_path != self.__state["select_request"]:
            return self.__handle_select_sources

        if response != 0:
            return None

        opts = {
            "handle_token": GLib.Variant("s", self.__generate_token()),
        }
        self.__state["start_request"] = self.__screencast.Start(self.__state["session_handle"], "s", opts)
        return self.__handle_start

    def __handle_start(self, object_path, response, results):
        if object_path != self.__state["start_request"]:
            return self.__handle_start

        if response != 0:
            return None

        streams = results.get("streams", [])
        if not streams:
            return None

        first = streams[0]
        node_id = first[0]
        self.__state["node_id"] = int(node_id)

    def __handle_response(self, sender, object_path, iface, signal, params):
        if iface != "org.freedesktop.portal.Request" or signal != "Response":
            return

        response, results = params
        self.__handler = self.__handler(object_path, response, results)

        if self.__handler is None:
            self.__loop.quit()

    def __start_screencast(self):
        create_opts = {
            "handle_token": GLib.Variant("s", self.__generate_token()),
            "session_handle_token": GLib.Variant("s", self.__generate_token()),
        }
        self.__state["create_request"] = self.__screencast.CreateSession(create_opts)
        self.__handler = self.__handle_create_session
        self.__loop.run()

        return self.__state["node_id"]

    def generate_capture_object(self) -> cv.VideoCapture:
        node_id = self.__start_screencast()
        pipeline = (
            f"pipewiresrc path={node_id} ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=1")
        capture = cv.VideoCapture(pipeline, cv.CAP_GSTREAMER)
        if not capture.isOpened():
            raise Exception(f"Could not open capture to node: {node_id}")
        return capture
