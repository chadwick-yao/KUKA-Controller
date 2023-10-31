import time
import numpy as np
from codebase.real_world.iiwaPy3 import iiwaPy3
from codebase.real_world.spacemouse import SpaceMouse


def test_getters(client: iiwaPy3):
    try:
        print(f"End effector position and orientation is: {client.getEEFPos()}")
        time.sleep(0.1)

        print(f"Forces acting at end effector are: {client.getEEF_Force()}")
        time.sleep(0.1)

        print(
            f"Cartezian position (X,Y,Z) of end effector: {client.getEEFCartizianPosition()}"
        )
        time.sleep(0.1)

        print(f"Moment at end effector: {client.getEEF_Moment()}")
        time.sleep(0.1)

        print(f"Joints positions: {client.getJointsPos()}")
        time.sleep(0.1)

        print(f"External torques at the joitns: {client.getJointsExternalTorques()}")
        time.sleep(0.1)

        print(f"Measured torques at the joints: {client.getJointsMeasuredTorques()}")
        time.sleep(0.1)

        print(f"Measured torque at joint 5: {client.getMeasuredTorqueAtJoint(5)}")
        time.sleep(0.1)
        print(
            f"Rotation of EEF, fixed rotation angles (X,Y,Z): {client.getEEFCartizianOrientation()}"
        )
        time.sleep(0.1)

        print(
            f"Joints positions has been streamed external torques are: {client.getEEFCartizianOrientation()}"
        )
        time.sleep(0.1)
    except:
        raise RuntimeError

    client.close()


if __name__ == "__main__":
    REMOTER = iiwaPy3(
        host="172.31.1.147",
        port=30001,
        trans=(0, 0, 0, 0, 0, 0),
    )

    test_getters(REMOTER)

    REMOTER.reset_initial_state()

    pos_sensitivity = 0.1
    rot_sensitivity = 0.1
    REMOTER.realTime_startDirectServoCartesian()
    with SpaceMouse(max_value=300) as DEVICE:
        action = DEVICE.get_motion_state_transformed()
        current_eef_pos = REMOTER.getEEFPos()
        next_eef_pos = np.zeros_like(current_eef_pos)

        next_eef_pos[:3] = current_eef_pos[:3] + action[:3] * pos_sensitivity
        next_eef_pos[3:] = current_eef_pos[3:] + action[3:] * rot_sensitivity

        DEVICE.sendEEfPosition(next_eef_pos)

        time.sleep(1 / 100)
    REMOTER.realTime_stopDirectServoCartesian()

    REMOTER.reset_initial_state()
    REMOTER.close()
