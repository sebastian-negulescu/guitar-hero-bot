import asyncio
import threading
import uinput

notes_to_keys = [uinput.KEY_A,
                 uinput.KEY_S, uinput.KEY_D,
                 uinput.KEY_F, uinput.KEY_G]
strum_key = uinput.KEY_J


async def tap_key(device, key, held_notes, delay=0.025):
    for note in held_notes:
        device.emit(note, 1)

    device.emit(key, 1)
    await asyncio.sleep(delay)
    device.emit(key, 0)

    for note in held_notes:
        device.emit(note, 0)

    held_notes.clear()


class Guitar:
    def __init__(self):
        self.__device = uinput.Device([strum_key] + notes_to_keys + [uinput.KEY_SPACE, uinput.KEY_ENTER, uinput.KEY_ESC],
                                      name="Virtual Dolphin Controller",
                                      vendor=0x045e,
                                      product=0x028e,
                                      bustype=0x03)

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

        self.__held_notes = set()

    def __del__(self):
        self.__device.destroy()

        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.loop.close()

    def strum(self):
        asyncio.run_coroutine_threadsafe(
            tap_key(self.__device, strum_key, self.__held_notes),
            self.loop)

    def press_note(self, note):
        key = notes_to_keys[note]
        if key not in self.__held_notes:
            self.__held_notes.add(key)
