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

    def connect(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to {self.host}:{self.port}")
        except socket.error as e:
            logger.error(f"Connection failed: {e}")

        # Update the transform of the TCP if one is specified
        if all(num == 0 for num in self.trans):
            logger.info("No TCP transform in Flange Frame is defined.")
            logger.info(
                f"The following (default) TCP transform is utilized: {self.trans}"
            )
            return

        logger.info("Trying to mount the following TCP transform:")
        string_tuple = (
            "x (mm)",
            "y (mm)",
            "z (mm)",
            "alfa (rad)",
            "beta (rad)",
            "gamma (rad)",
        )

        for i in range(6):
            print(string_tuple[i] + ": " + str(self.trans[i]))

        da_message = "TFtrans_" + "_".join(map(str, self.trans)) + "\n"
        self.send(da_message)
        return_ack_nack = self.receive()

        if "done" in return_ack_nack:
            logger.info("Specified TCP transform mounted successfully")
        else:
            raise RuntimeError("Could not mount the specified TCP")

    def close(self):
        assert self.sock, "No connection is detected."

        self.send("end\n")
        time.sleep(1)
        self.sock.close()

        logger.info(f"Connection to {self.host}:{self.port} was destroyed.")

    def send(self, data: str):
        assert self.sock, "No connection is detected."

        try:
            self.sock.send(data.encode("utf-8"))
        except socket.error as e:
            logger.error(f"Send error: {e}.")

    def receive(self, buffer_size=1024):
        assert self.sock, "No connection is detected."

        try:
            return self.sock.recv(buffer_size).decode("utf-8")
        except socket.error as e:
            logger.error(f"Recv error: {e}.")
