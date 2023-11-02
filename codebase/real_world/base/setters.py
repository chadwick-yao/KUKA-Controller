import sys
from socket import socket
import logging
from typing import Tuple
from codebase.real_world.base.base_client import BaseClient

logger = logging.getLogger(__name__)


class Setters(BaseClient):
    def __init__(self, host: str, port: int, trans: Tuple, sock: socket) -> None:
        super().__init__(host, port, trans)

        self.set_socket(sock)
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

        self.send(command)
        message = self.receive()

        logger.info(f"Got message: {message}, after doing {command}")
        sys.stdout.flush()
