import sys
import os
import pathlib
ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)

import api.sim as sim
import time

def control_UR10():
    sim.simxFinish(-1) # just in case, close all opened connections
    clientID = sim.simxStart('127.0.0.1',19999,True,True,5000,5) # Connect to CoppeliaSim
    if clientID != -1:
        print ('Connected to remote API server')

        res, objs = sim.simxGetObjects(clientID, sim.sim_handle_all, sim.simx_opmode_blocking)
        if res == sim.simx_return_ok:
            print ('Number of objects in the scene: ',len(objs))
        else:
            print ('Remote API function call returned with error code: ',res)

        sim_ret, target_handle = sim.simxGetObjectHandle(clientID, 'target', sim.simx_opmode_blocking)

        while True:
            sim_ret, target_ori = sim.simxGetObjectOrientation(clientID, target_handle, -1, sim.simx_opmode_blocking)
            sim_ret, target_pos = sim.simxGetObjectPosition(clientID, target_handle, -1, sim.simx_opmode_blocking)
            print(target_ori, target_pos)
            time.sleep(0.1)


        # Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
        sim.simxGetPingTime(clientID)

        # Now close the connection to CoppeliaSim:
        sim.simxFinish(clientID)
    else:
        print ('Failed connecting to remote API server')
    print ('Program ended')

control_UR10()