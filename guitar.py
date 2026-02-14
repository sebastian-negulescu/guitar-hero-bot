import asyncio
import uinput

notes_to_keys = [uinput.KEY_A,
                 uinput.KEY_S, uinput.KEY_D,
                 uinput.KEY_F, uinput.KEY_G]
strum_key = uinput.KEY_J


async def tap_key(device, key, delay=0.025):
    device.emit(key, 1)
    await asyncio.sleep(delay)
    device.emit(key, 0)


class Guitar:
    def __init__(self):
        self.__device = uinput.Device([strum_key] + notes_to_keys + [uinput.KEY_SPACE, uinput.KEY_ENTER, uinput.KEY_ESC],
                                      name="Virtual Dolphin Controller",
                                      vendor=0x045e,
                                      product=0x028e,
                                      bustype=0x03)
        self.__held_notes = set()

    def __del__(self):
        self.__device.destroy()

    def strum(self):
        asyncio.run(tap_key(self.__device, strum_key))
        for key in self.__held_notes:
            self.__device.emit(key, 0)
        self.__held_notes.clear()

    def press_note(self, note):
        key = notes_to_keys[note]
        if key not in self.__held_notes:
            self.__device.emit(key, 1)
            self.__held_notes.add(key)
