from utils.data_utils import String2Double
from socket import socket


class Getters(object):
    def __init__(self, sock: socket, iter=5) -> None:
        self.sock = sock
        self.retry = iter

    def _get_data(self, command, size):
        command = command + "\n"
        for _ in range(self.retry):
            self.sock.send(command.encode("utf-8"))
            message = self.sock.recv(1024).decode("utf-8")
            data = String2Double(message=message, size=size)
            if data:
                return data
        return []

    def get_EEF_pos(self):
        return self._get_data("Eef_pos", 6)

    def get_EEF_force(self):
        return self._get_data("Eef_force", 3)

    def get_EEF_CartizianPos(self):
        return self._get_data("Eef_pos", 3)

    def get_EEF_moment(self):
        return self._get_data("Eef_moment", 3)

    def get_JointPos(self):
        return self._get_data("getJointsPositions", 7)

    def get_Joints_ExternalTorques(self):
        return self._get_data("Torques_ext_J", 7)

    def get_Joints_MeasuredTorques(self):
        return self._get_data("Torques_m_J", 7)

    def get_MeasuredTorques_at_Joint(self, idx: int):
        assert 0 <= idx < 7, "index must be in [0,7)."

        result = self._get_data("Torques_m_J", 7)
        if result:
            return result[idx]
        return []

    def get_EEF_CartizianOrientation(self):
        result = self._get_data("Eef_pos", 6)
        if result:
            return result[3:]
        return []

    def get_pinState(self, idx: int):
        assert idx in [
            3,
            4,
            10,
            13,
            16,
        ], "Unsupported pins, ensure pins are in [3, 4, 10, 13, 16]."

        return self._get_data("getPin" + str(idx), 1)
