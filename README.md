# Guitar Hero Bot

using `Python 3.13.5`
requires `numpy, pydbus, python-uinput, evdev`

also requires `opencv-python` built with GStreamer support

## Getting uinput working

Need to load uinput module and ensure you are a part of the input group.

`modprobe -i uinput`

`usermod -a -G input <user>`

