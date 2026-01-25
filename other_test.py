import os
import cv2

# FORCE the environment variables that Hyprland sometimes "forgets" to give D-Bus
# This tells the portal "I am running on Hyprland, please show the popup here."
os.environ["XDG_SESSION_TYPE"] = "wayland"
os.environ["XDG_CURRENT_DESKTOP"] = "Hyprland"
os.environ["WAYLAND_DISPLAY"] = os.environ.get("WAYLAND_DISPLAY", "wayland-1")

token = input("token: ")


def capture_screen_pipewire():
    # 1. Define the GStreamer pipeline
    # pipewiresrc: Triggers D-Bus request to xdg-desktop-portal
    # videoconvert: Ensures color format compatibility with OpenCV (BGR)
    # appsink: The bridge to OpenCV
    gst_pipeline = (
        f"pipewiresrc path={token} ! "
        "videoconvert ! "
        "autovideosink"
    )

    print("Requesting screen access via D-Bus... (Check for OS popup)")
    # 2. Initialize VideoCapture with GStreamer backend
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Error: Could not open stream. Did you allow access in the popup?")
        return

    print("Capture started. Press 'q' to quit.")

    try:
        while True:
            # 3. Read frame
            ret, frame = cap.read()
            if not ret:
                print("Stream ended.")
                break

            # --- OpenCV Processing Here ---
            # Example: Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ------------------------------

            cv2.imshow("OpenCV PipeWire Capture", gray)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_screen_pipewire()
