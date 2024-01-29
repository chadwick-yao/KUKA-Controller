import numpy as np
from typing import Tuple, Union
import logging
import socket
import time
import copy
import enum
import os
from termcolor import colored, cprint
from utils.data_utils import pose_euler2quat

FORMAT = "[%(asctime)s][%(levelname)s]: %(message)s"
logging.basicConfig(
    level=logging.INFO, format=FORMAT, handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

from codebase.real_world.base.PTP import PTP
from codebase.real_world.base.getters import Getters
from codebase.real_world.base.senders import Senders
from codebase.real_world.base.setters import Setters
from codebase.real_world.base.RealTime import RealTime
from codebase.real_world.base.base_client import BaseClient

import multiprocessing as mp
from typing import Optional, Tuple, List
from multiprocessing.managers import SharedMemoryManager
from codebase.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from codebase.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

np.set_printoptions(precision=2, suppress=True, linewidth=100)


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    RESET = 3


class IIWAPositionalController(BaseClient, mp.Process):
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        receive_keys: Optional[List],
        host: str = "172.31.1.147",
        port: int = 30001,
        trans: Tuple = (0, 0, 0, 0, 0, 0),
        frequency: int = 100,
        max_pos_speed: float = 32,
        max_rot_speed: float = 0.5,
        launch_timeout: int = 3,
        soft_real_time: bool = False,
        verbose: bool = False,
        get_max_k: int = 128,
        use_quat: bool = True,
    ) -> None:
        """
        frequency: socket connection frequency
        receive_keys: data definition
        real max_pos_speed: mm/s
        real max_rot_speed: rad/s
        soft_real_time: enables round-robin scheduling and real-time priority reuqires running scripts before hand
        """
        # super init
        BaseClient.__init__(self, host, port, trans)
        mp.Process.__init__(self, name="IIWAPositionalController")

        # robot connection
        self.connect()
        self.setter = Setters(host, port, trans, self.sock)
        self.getter = Getters(host, port, trans, self.sock)
        self.sender = Senders(host, port, trans, self.sock)
        self.rtl = RealTime(host, port, trans, self.sock)
        self.ptp = PTP(host, port, trans, self.sock)
        self.TCPtrans = trans

        self.frequency = frequency
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        self.get_max_k = get_max_k
        self.use_quat = use_quat

        # init pose (PTP)
        self.reset_initial_state()
        self.init_eef_pose = self.getEEFPos()

        # build input queue
        example = {
            "cmd": Command.SERVOL.value,
            "target_pose": np.zeros((7,), dtype=np.float64)
            if self.use_quat
            else np.zeros((6,), dtype=np.float64),
            "duration": 0.0,  # desired time to reach pose
            "target_time": 0.0,
        }

        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256,
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = ["EEFpos", "EEFrot", "Jpos"]

        example = dict()
        pose = np.array(getattr(self, "getEEFPos")())
        example[receive_keys[0]] = pose[:3]
        example[receive_keys[1]] = pose[3:]
        example[receive_keys[2]] = np.array(getattr(self, "getJointsPos")())
        example["robot_receive_timestamp"] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        self.ready_event = mp.Event()
        self.ready_servo = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            cprint(
                f"[IIWAPositionalController] Controller process spawned at {self.pid}",
                "white",
                "on_green",
            )

    def stop(self, wait=True):
        message = {
            "cmd": Command.STOP.value,
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        # assert self.is_alive()
        assert duration >= (1 / self.frequency)
        pose = np.array(pose)
        assert pose.shape == (6,) or pose.shape == (7,)

        message = {
            "cmd": Command.SERVOL.value,
            "target_pose": pose,
            "duration": duration,
        }
        self.input_queue.put(message)

    def reset_robot(self):
        message = {
            "cmd": Command.RESET.value,
        }
        self.input_queue.put(message)

    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pose": pose,
            "target_time": target_time,
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def run(self):
        cprint("Now Robot threading is running!", "yellow")
        if self.soft_real_time:
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))

        try:
            # main loop
            dt = 1.0 / self.frequency
            curr_pose = self.getEEFPos()

            target_pose = copy.deepcopy(curr_pose)
            if self.use_quat:
                target_pose = pose_euler2quat(target_pose)
            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t], poses=[target_pose]
            )

            iter_idx = 0
            keep_running = True

            self.realTime_startDirectServoCartesian()
            self.ready_servo.set()

            while keep_running:
                t_start = time.perf_counter()

                t_now = time.monotonic()
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print("extrapolate", diff)
                pose_command = pose_interp(t_now)
                if self.use_quat:
                    pose_command = pose_euler2quat(pose_command)
                # print(pose_command)
                # update robot state
                state = dict()
                ActualPose = np.array(
                    getattr(self, "sendEEfPositionGetActualEEFpos")(pose_command)
                )
                state["EEFpos"] = ActualPose[:3]
                state["EEFrot"] = ActualPose[3:]
                state["Jpos"] = np.array(
                    getattr(self, "sendEEfPositionGetActualJpos")(pose_command)
                )

                state["robot_receive_timestamp"] = time.time()
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command["cmd"]

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity
                        # and cause jittery robot behavior.
                        target_pose = command["target_pose"]
                        duration = float(command["duration"])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            cprint(
                                f"[IIWAPositionalController] New pose target:{target_pose} duration:{duration}s",
                                "red",
                            )
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command["target_pose"]
                        target_time = float(command["target_time"])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time,
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.RESET.value:
                        self.realTime_stopDirectServoCartesian()
                        self.reset_initial_state()
                        target_pose = self.getEEFPos()
                        curr_time = time.monotonic()
                        self.init_eef_pose = target_pose
                        if self.use_quat:
                            target_pose = pose_euler2quat(target_pose)
                        if self.verbose:
                            cprint(
                                f"After resetting, target pose -> {target_pose}",
                                color="blue",
                            )
                        self.realTime_startDirectServoCartesian()

                        del pose_interp
                        pose_interp = PoseTrajectoryInterpolator(
                            times=[curr_time], poses=[target_pose]
                        )
                        break
                    else:
                        keep_running = False
                        break
                t_end = time.perf_counter()
                if t_end - t_start < dt:
                    time.sleep(dt - t_end + t_start)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    print("IIWA ready event is set!!!")
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(
                        f"[IIWAPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}"
                    )

        finally:
            # terminate
            self.realTime_stopDirectServoCartesian()
            self.reset_initial_state()
            self.close()
            self.ready_event.set()

    def connect(self):
        try:
            self.set_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            self.sock.settimeout(15.0)
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

    def reset_initial_state(self):
        init_jpos = [0, np.pi * 30 / 180, 0, -np.pi * 80 / 180, 0, np.pi * 70 / 180, 0]
        # joint_deviations = np.random.uniform(
        #     low=-1.0 * 3.14 / 180, high=1.0 * 3.14 / 180, size=7
        # )
        # init_jpos += joint_deviations
        init_vel = [0.2]
        cprint(f"Reset to jpos: {init_jpos}", "blue")
        self.movePTPJointSpace(jpos=init_jpos, relVel=init_vel)

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

    def movePTPLineEEF(self, pos, vel):
        self.ptp.movePTPLineEEF(pos, vel)

    """
    Cartesian linear  motion
    """

    def movePTPLineEEF(self, pos, vel):
        self.ptp.movePTPLineEEF(pos, vel)

    def movePTPLineEefRelBase(self, pos, vel):
        self.ptp.movePTPLineEefRelBase(pos, vel)

    def movePTPLineEefRelEef(self, pos, vel):
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

    def realTime_startDirectServoCartesian(self):
        self.rtl.realTime_startDirectServoCartesian()

    def realTime_stopDirectServoCartesian(self):
        self.rtl.realTime_stopDirectServoCartesian()

    def realTime_stopDirectServoJoints(self):
        self.rtl.realTime_stopDirectServoJoints()

    def realTime_startDirectServoJoints(self):
        self.rtl.realTime_startDirectServoJoints()

    def realTime_startImpedanceJoints(
        self, weightOfTool, cOMx, cOMy, cOMz, cStiness, rStifness, nStifness
    ):
        self.rtl.realTime_startImpedanceJoints(
            weightOfTool, cOMx, cOMy, cOMz, cStiness, rStifness, nStifness
        )

    # Joint space servo command
    def sendJointsPositionsGetMTorque(self, x):
        return self.sender.sendJointsPositionsGetMTorque(x)

    def sendJointsPositionsGetExTorque(self, x):
        return self.sender.sendJointsPositionsGetExTorque(x)

    def sendJointsPositionsGetActualEEFpos(self, x):
        return self.sender.sendJointsPositionsGetActualEEFpos(x)

    def sendJointsPositionsGetEEF_Force_rel_EEF(self, x):
        return self.sender.sendJointsPositionsGetEEF_Force_rel_EEF(x)

    def sendJointsPositionsGetActualJpos(self, x):
        return self.sender.sendJointsPositionsGetActualJpos(x)

    # Crtesian space servo command
    def sendEEfPosition(self, x):
        self.sender.sendEEfPosition(x)

    def sendEEfPositionGetExTorque(self, x):
        return self.sender.sendEEfPositionExTorque(x)

    def sendEEfPositionGetActualEEFpos(self, x):
        return self.sender.sendEEfPositionGetActualEEFpos(x)

    def sendEEfPositionGetActualJpos(self, x):
        return self.sender.sendEEfPositionGetActualJpos(x)

    def sendEEfPositionGetEEF_Force_rel_EEF(self, x):
        return self.sender.sendEEfPositionGetEEF_Force_rel_EEF(x)

    def sendEEfPositionGetMTorque(self, x):
        return self.sender.sendEEfPositionMTorque(x)

    # getters
    def getEEFPos(self):
        return self.getter.get_EEF_pos()

    def getEEF_Force(self):
        return self.getter.get_EEF_force()

    def getEEFCartizianPosition(self):
        return self.getter.get_EEF_CartizianPos()

    def getEEF_Moment(self):
        return self.getter.get_EEF_moment()

    def getJointsPos(self):
        return self.getter.get_JointPos()

    def getJointsExternalTorques(self):
        return self.getter.get_Joints_ExternalTorques()

    def getJointsMeasuredTorques(self):
        return self.getter.get_Joints_MeasuredTorques()

    def getMeasuredTorqueAtJoint(self, x):
        return self.getter.get_MeasuredTorques_at_Joint(x)

    def getEEFCartizianOrientation(self):
        return self.getter.get_EEF_CartizianOrientation()

    # get pin states
    def getPin3State(self):
        return self.getter.get_pinState(3)

    def getPin10State(self):
        return self.getter.get_pinState(10)

    def getPin13State(self):
        return self.getter.get_pinState(13)

    def getPin16State(self):
        return self.getter.get_pinState(16)

    # setters
    def set_OnOff(self, cmd):
        self.setter.set_OnOff(cmd)
