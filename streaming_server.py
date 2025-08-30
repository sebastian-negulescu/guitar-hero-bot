import asyncio
import os
import logging
import simpleobsws

from pyrtmp import StreamClosedException
from pyrtmp.flv import FLVFileWriter, FLVMediaType
from pyrtmp.session_manager import SessionManager
from pyrtmp.rtmp import SimpleRTMPController, RTMPProtocol, SimpleRTMPServer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

sensitive_poll_delay_s = 1
poll_delay_s = 1


class RTMP2FLVController(SimpleRTMPController):

    def __init__(self, output_filepath: str):
        self.output_filepath = output_filepath
        super().__init__()

    async def on_ns_publish(self, session, message) -> None:
        session.state = FLVFileWriter(output=self.output_filepath)
        await super().on_ns_publish(session, message)

    async def on_metadata(self, session, message) -> None:
        session.state.write(0, message.to_raw_meta(), FLVMediaType.OBJECT)
        await super().on_metadata(session, message)

    async def on_video_message(self, session, message) -> None:
        session.state.write(message.timestamp, message.payload, FLVMediaType.VIDEO)
        await super().on_video_message(session, message)

    async def on_audio_message(self, session, message) -> None:
        session.state.write(message.timestamp, message.payload, FLVMediaType.AUDIO)
        await super().on_audio_message(session, message)

    async def on_stream_closed(self, session: SessionManager, exception: StreamClosedException) -> None:
        session.state.close()
        await super().on_stream_closed(session, exception)


class SimpleServer(SimpleRTMPServer):

    def __init__(self, output_filepath: str):
        self.output_filepath = output_filepath
        super().__init__()

    async def create(self, host: str, port: int):
        loop = asyncio.get_event_loop()
        self.server = await loop.create_server(
            lambda: RTMPProtocol(controller=RTMP2FLVController(self.output_filepath)),
            host=host,
            port=port,
        )


async def create_server():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server = SimpleServer(output_filepath=os.path.join(current_dir, "stream.flv"))
    await server.create(host='0.0.0.0', port=1935)
    await server.start()
    return server


async def run_server():
    server = await create_server()
    try:
        await server.wait_closed()
    except asyncio.CancelledError:
        await server.stop()


async def create_client():
    with open("ws_password.txt", "r") as password_file:
        password = password_file.read().strip()
    client = simpleobsws.WebSocketClient(url="ws://localhost:4455", password=password)
    await client.connect()
    await client.wait_until_identified()
    return client


async def start_stream(client):
    start_stream_request = simpleobsws.Request("StartStream")
    ret = await client.call(start_stream_request)
    if not ret.ok():
        await client.disconnect()
        return

    ret = await client.call(simpleobsws.Request("GetStreamStatus"))
    while not ret.responseData["outputActive"]:
        await asyncio.sleep(poll_delay_s)
        ret = await client.call(simpleobsws.Request("GetStreamStatus"))


async def stop_stream(client):
    ret = await client.call(simpleobsws.Request("StopStream"))
    if not ret.ok():
        await client.disconnect()

    ret = await client.call(simpleobsws.Request("GetStreamStatus"))
    while ret.responseData["outputActive"]:
        await asyncio.sleep(sensitive_poll_delay_s)
        ret = await client.call(simpleobsws.Request("GetStreamStatus"))

    await client.disconnect()


async def run_stream(start_event, stop_event):
    client = await create_client()
    await start_stream(client)
    start_event.set()
    while not stop_event.is_set():
        await asyncio.sleep(poll_delay_s)
    await stop_stream(client)


async def run_service(start_event, stop_event):
    async with asyncio.TaskGroup() as tg:
        server_task = tg.create_task(run_server())
        stream_task = tg.create_task(run_stream(start_event, stop_event))

        await stream_task
        server_task.cancel()


def run_screen_record(start_event, stop_event):
    asyncio.run(run_service(start_event, stop_event))
