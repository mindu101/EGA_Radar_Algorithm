from .serial_comm import RefereeSerialManager
from .serial_protocol import SentrySelfPositionMessage, RobotStatusMessage
from .serial_protocol import Radar2SentryMessage
from .serial_protocol import MsgID, SubCmdID


def status_message_decode_func(cmd_id, data):

    if cmd_id == MsgID.ROBOT_DATA.value:
        message = RobotStatusMessage.from_bytes(data)
        print(f"[STATUS] Robot Status: {message}")


def sentry_self_position_message_decode_func(cmd_id, data):
    if cmd_id == MsgID.INTERACTIVE_DATA.value:
        message = SentrySelfPositionMessage.from_bytes(data)
        print(f"[SENTRY_SELF_POSITION] Sentry self position message: {message}")


if __name__ == "__main__":
    serial_manager = RefereeSerialManager(port="/dev/ttyUSB0", baudrate=115200)
    serial_manager.bind(MsgID.ROBOT_DATA.value, status_message_decode_func)
    serial_manager.bind(MsgID.INTERACTIVE_DATA.value, sentry_self_position_message_decode_func)
    serial_manager.start()
    import time

    while True:
        # serial_manager.summarize()
        sentry_msg = Radar2SentryMessage(
            is_blue=True,
            targetable_flags=0xFF,
        )
        serial_manager.summarize()
        serial_manager.tx(sentry_msg.pack())
        time.sleep(1.0)
