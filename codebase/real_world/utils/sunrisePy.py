from codebase.real_world.utils.mySock import mySock
from codebase.real_world.utils.getters import Getters
from codebase.real_world.utils.setters import Setters
from codebase.real_world.utils.RealTime import RealTime
from codebase.real_world.utils.senders import Senders
from codebase.real_world.utils.PTP import PTP

class sunrisePy(object):
    getters = 0
    realtime = 0
    gnerealPorpuse = 0
    
    def __init__(self, ip):
        port=30001
        self.soc = mySock(remote_host=ip, remote_port=port)
        self.set = Setters(self.soc)
        self.get = Getters(self.soc)
        self.sender = Senders(self.soc)
        self.rtl = RealTime(self.soc)
        self.ptp = PTP(self.soc)
  
    def close(self):
        self.soc.close()
    
    def send(self,data):
        self.soc.send(data)
# PTP motion
    """
    Joint space motion
    """
    def movePTPJointSpace(self, jpos, relVel):
        self.ptp.movePTPJointSpace(jpos, relVel)
    
    def movePTPHomeJointSpace(self, relVel):
        self.ptp.movePTPHomeJointSpace(relVel)
        
    def movePTPTransportPositionJointSpace(self, relVel):
        self.ptp.movePTPTransportPositionJointSpace(relVel)
    """
    Cartesian linear  motion
    """
    def movePTPLineEEF(self, pos, vel):
        self.ptp.movePTPLineEEF(pos, vel)
        
    def movePTPLineEefRelBase(self,pos, vel):
        self.ptp.movePTPLineEefRelBase(pos, vel)
        
    def movePTPLineEefRelEef(self,pos, vel):
        self.ptp.movePTPLineEefRelEef(pos, vel)
    """
    Circular motion
    """        
    def movePTPCirc1OrintationInter(self, f1, f2, vel):
        self.ptp.movePTPCirc1OrintationInter(f1, f2, vel)
        
    def movePTPArcYZ_AC(self, theta, c, vel):
        self.ptp.movePTPArcYZ_AC(theta, c, vel)
        
    def movePTPArcXZ_AC(self, theta, c, vel):
        self.ptp.movePTPArcXZ_AC(theta, c, vel)
        
    def movePTPArcXY_AC(self, theta, c, vel):
        self.ptp.movePTPArcXY_AC(theta, c, vel)
        
    def movePTPArc_AC(self, theta, c, k, vel):
        self.ptp.movePTPArc_AC(theta, c, k, vel)
        
# realtime motion control
    def realTime_stopImpedanceJoints(self):
        self.rtl.realTime_stopImpedanceJoints()
        
    def realTime_stopDirectServoJoints(self):
        self.rtl.realTime_stopDirectServoJoints()
        
    def realTime_startDirectServoJoints(self):  
        self.rtl.realTime_startDirectServoJoints()
    
    def realTime_startImpedanceJoints(self, weightOfTool, cOMx, cOMy, cOMz, cStiness, rStifness, nStifness):
        self.rtl.realTime_startImpedanceJoints(weightOfTool, cOMx, cOMy, cOMz, cStiness, rStifness, nStifness)
    
    def sendJointsPositions(self, x):
        self.sender.sendJointsPositions(x)
        
    def sendJointsPositionsGetMTorque(self, x): 
        return self.sender.sendJointsPositionsGetMTorque(x)
        
    def sendJointsPositionsGetExTorque(self, x):
        return self.sender.sendJointsPositionsGetExTorque(x)
        
    def sendJointsPositionsGetActualJpos(self, x):
        return self.sender.sendJointsPositionsGetActualJpos(x)
        
# getters
    def getEEFPos(self):
        return self.get.get_EEF_pos()
    
    def getEEF_Force(self):
        return self.get.get_EEF_force()
        
    def getEEFCartizianPosition(self):
        return self.get.get_EEF_CartizianPos()
        
    def getEEF_Moment(self):
        return self.get.get_EEF_moment()
        
    def getJointsPos(self):
        return self.get.get_JointPos()
        
    def getJointsExternalTorques(self):
        return self.get.get_Joints_ExternalTorques()
        
    def getJointsMeasuredTorques(self):
        return self.get.get_Joints_MeasuredTorques()
        
    def getMeasuredTorqueAtJoint(self,x):
        return self.get.get_MeasuredTorques_at_Joint(x)
        
    def getEEFCartizianOrientation(self):
        return self.get.get_EEF_CartizianOrientation()
        
# get pin states 
    def getPin3State(self):
        return self.get.get_pinState(3)
    
    def getPin10State(self):
        return self.get.get_pinState(10)
    
    def getPin13State(self):
        return self.get.get_pinState(13)
        
    def getPin16State(self):
        return self.get.get_pinState(16)
        
# setters
    def set_OnOff(self, cmd):
        self.set.set_OnOff(cmd)