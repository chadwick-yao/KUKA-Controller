import sys
from codebase.real_world.utils.mySock import mySock

class Setters(object):

    def __init__(self,mysoc: mySock):
        self.mysoc=mysoc
        self.cmd_list = [
            'blueOff', 'blueOn', 'pin1off', 'pin1on',
            'pin2off', 'pin2on', 'pin11off', 'pin11on',
            'pin12off', 'pin12on'
        ]

    def send(self,data):
        data = data+'\n'
        self.mysoc.send(data)
        message = self.mysoc.receive()
        print(message)
        sys.stdout.flush()

    def set_OnOff(self, cmd):
        assert cmd in self.cmd_list, "Unsupported Commonds."

        self.send(cmd)