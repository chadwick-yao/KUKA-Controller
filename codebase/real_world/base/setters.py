import sys
from socket import socket
import logging

logger = logging.getLogger(__name__)


class Setters(object):
    def __init__(self, sock: socket):
        self.sock = sock
        self.cmd_list = [
            "blueOff",
            "blueOn",
            "pin1off",
            "pin1on",
            "pin2off",
            "pin2on",
            "pin11off",
            "pin11on",
            "pin12off",
            "pin12on",
        ]

    def set_OnOff(self, command: str):
        assert command in self.cmd_list, "Unsupported Commonds."

        command = command + "\n"

        self.sock.send(command.encode("utf-8"))
        message = self.sock.recv(1024).decode("utf-8")

        logger.info(f"Got message: {message}, after doing {command}")
        sys.stdout.flush()
