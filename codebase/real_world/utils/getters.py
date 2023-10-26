from common.data_utils import String2Double

class Getters(object):

    def __init__(self, mysoc, iter) -> None:
        self.mysoc = mysoc
        self.numOfIterations = iter
    
    def _send_array(self, data, size):
        data = data + "\n"
        self.mysoc.send(data)
        message = self.mysoc.receive()
        
        return String2Double(message=message, size=size)

    def _send_element(self, data):
        data = data + "\n"
        self.mysoc.send(data)
        message = self.mysoc.receive()

        return float(message)

    def _get_data(self, command, size):
        for _ in range(self.numOfIterations):
            data = self._send_array(command, size)
            if data:
                return data
        return []

    def get_EEF_pos(self):
        return self._get_data('Eef_pos', 6)

    def get_EEF_force(self):
        return self._get_data('Eef_force', 3)

    def get_EEF_CartizianPos(self):
        return self._get_data('Eef_pos', 3)

    def get_EEF_moment(self):
        return self._get_data('Eef_moment', 3)
    
    def get_JointPos(self):
        return self._get_data('getJointsPositions', 7)

    def get_Joints_ExternalTorques(self):
        return self._get_data('Torques_ext_J', 7)

    def get_Joints_MeasuredTorques(self):
        return self._get_data('Torques_m_J', 7)

    def get_MeasuredTorques_at_Joint(self, idx: int):
        assert 0 <= idx <7, "index must be in [0,7)."

        result = self._get_data('Torques_m_J', 7)
        if result:
            return result[idx]
        return []
    
    def get_EEF_CartizianOrientation(self):
        result = self._get_data('Eef_pos', 6)
        if result:
            return result[3:]
        return []

    def get_pinState(self, idx: int):
        assert idx in [3, 4, 10, 13, 16], "Unsupported pins to check."

        return self._send_element('getPin' + str(idx))