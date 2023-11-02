from typing import Tuple, Union, Optional
from abc import ABCMeta, abstractmethod
import logging
import socket
import time

FORMAT = "[%(asctime)s][%(levelname)s]: %(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


class BaseClient(metaclass=ABCMeta):
    def __init__(
        self,
        host: str,
        port: int,
        trans: Tuple = (0, 0, 0, 0, 0, 0),
    ) -> None:
        self.host = host
        self.port = port
        self.trans = trans
        self.sock = None

    @property
    def remote_info(self):
        return (self.host, self.port)

    def set_socket(self, sock: socket.socket):
        self.sock = sock

    def close(self):
        assert self.sock, "No connection is detected."

        self.sock.send("end\n".encode("utf-8"))
        time.sleep(1)
        self.sock.close()

        logger.info(f"Connection to {self.host}:{self.port} was destroyed.")

    def send(self, data: str):
        assert self.sock, "No connection is detected."

        try:
            self.sock.send(data.encode("utf-8"))
        except socket.error as e:
            self.close()
            logger.error(f"Send error: {e}.")

    def receive(self, buffer_size=1024):
        assert self.sock, "No connection is detected."

        try:
            return self.sock.recv(buffer_size).decode("utf-8")
        except socket.error as e:
            self.close()
            logger.error(f"Recv error: {e}.")
