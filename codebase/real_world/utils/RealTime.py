import time

class RealTime:
    def __init__(self, mysoc):
        self.mysoc = mysoc

    def send(self, data):
        data = data + '\n'
        self.mysoc.send(data)
        self.mysoc.receive()

    def realTime_startDirectServoJoints(self):
        theCommand = 'startDirectServoJoints'
        self.send(theCommand)
        time.sleep(0.3)

    def realTime_stopDirectServoJoints(self):
        theCommand = 'stopDirectServoJoints'
        self.send(theCommand)
        time.sleep(0.3)

    def realTime_startDirectServoCartesian(self):
        theCommand = 'startDirectServoCartesian'
        self.send(theCommand)
        time.sleep(0.3)

    def realTime_stopDirectServoCartesian(self):
        theCommand = 'stopDirectServoCartesian'
        self.send(theCommand)
        time.sleep(0.3)

    def realTime_startImpedanceJoints(self, weightOfTool, cOMx, cOMy, cOMz, cStiffness, rStiffness, nStiffness):
        theCommand = 'startSmartImpedanceJoints'
        theCommand = theCommand + '_' + str(weightOfTool)
        theCommand = theCommand + '_' + str(cOMx)
        theCommand = theCommand + '_' + str(cOMy)
        theCommand = theCommand + '_' + str(cOMz)
        theCommand = theCommand + '_' + str(cStiffness)
        theCommand = theCommand + '_' + str(rStiffness)
        theCommand = theCommand + '_' + str(nStiffness) + '_'
        self.send(theCommand)
        time.sleep(0.3)

    def realTime_stopImpedanceJoints(self):
        theCommand = 'stopImpedanceJoints'
        self.send(theCommand)
        time.sleep(0.3)