import math
from utils.data_utils import String2Double
from socket import socket


class Senders(object):
    def __init__(self, sock: socket) -> None:
        self.sock = sock

    def send(self, data: str):
        data = data + "\n"
        self.sock.send(data.encode("utf-8"))
        return self.sock.recv(1024).decode("utf-8")

    # EEF commond
    def _send_EEF_info(self, data, cmd, ret=False):
        assert len(data) == 6, "EEF position should be an array of 6 elements."

        num = 10000
        formatted_data = [
            str(math.ceil(value * num) / num) if i < 3 else str(value)
            for i, value in enumerate(data)
        ]

        buff = cmd + "_".join(formatted_data)

        result = self.send(buff)
        if ret:
            return result

    def sendEEfPosition(self, x):
        self._send_EEF_info(data=x, cmd="DcSeCarW_", ret=False)

    def sendEEfPositions(self, x):
        self._send_EEF_info(data=x, cmd="cArtixanPosition_", ret=False)

    def sendEEfPositionExTorque(self, x):
        return String2Double(
            self._send_EEF_info(data=x, cmd="DcSeCarExT_", ret=True), 7
        )

    def sendEEfPositionGetActualEEFpos(self, x):
        return String2Double(
            self._send_EEF_info(data=x, cmd="DcSeCarEEfP_", ret=True), 6
        )

    def sendEEfPositionGetActualJpos(self, x):
        return String2Double(self._send_EEF_info(data=x, cmd="DcSeCarJP_", ret=True), 7)

    def sendEEfPositionGetEEF_Force_rel_EEF(self, x):
        return String2Double(
            self._send_EEF_info(data=x, cmd="DcSeCarEEfP_", ret=True), 6
        )

    def sendEEfPositionMTorque(self, x):
        return String2Double(self._send_EEF_info(data=x, cmd="DcSeCarMT_", ret=True), 7)

    def _send_Joints_info(self, data, cmd, ret=False):
        assert len(data) == 7, "Joints should be an array of 7 elements."

        num = 10000
        formatted_data = [str(math.ceil(value * num) / num) for value in data]

        buff = cmd + "_".join(formatted_data)
        result = self.send(buff)

        if ret:
            return result

    def sendJointsPositions(self, x):
        self._send_Joints_info(data=x, cmd="jp_", ret=False)

    def sendJointsPositionsGetMTorque(self, x):
        return String2Double(self._send_Joints_info(data=x, cmd="jpMT_", ret=True), 7)

    def sendJointsPositionsGetActualEEFpos(self, x):
        return String2Double(self._send_Joints_info(data=x, cmd="jpEEfP_", ret=True), 6)

    def sendJointsPositionsGetEEF_Force_rel_EEF(self, x):
        return String2Double(
            self._send_Joints_info(data=x, cmd="DcSeCarEEfFrelEEF_", ret=True), 6
        )

    def sendJointsPositionsGetExTorque(self, x):
        return String2Double(self._send_Joints_info(data=x, cmd="jpExT_", ret=True), 7)

    def sendJointsPositionsGetActualJpos(self, x):
        return String2Double(self._send_Joints_info(data=x, cmd="jpJP_", ret=True), 7)

    # functions for arc motion
    def sendCirc1FramePos(self, fpos):
        assert len(fpos) == 6, "EEF position should be an array of 6 elements."

        num = 10000

        cmd = "cArtixanPositionCirc1_"
        formatted_data = [str(math.ceil(value * num) / num) for value in fpos]

        buff = cmd + "_".join(formatted_data)
        self.send(buff)

    def sendCirc2FramePos(self, fpos):
        assert len(fpos) == 6, "EEF position should be an array of 6 elements."

        num = 10000

        cmd = "cArtixanPositionCirc2_"
        formatted_data = [str(math.ceil(value * num) / num) for value in fpos]

        buff = cmd + "_".join(formatted_data)
        self.send(buff)
