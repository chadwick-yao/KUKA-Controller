#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import enum
import numpy as np
from math import ceil
import multiprocessing as mp
from typing import Optional, List, Dict, Tuple, Union
from pymodbus.client.sync import ModbusSerialClient

import sys
import os
import pathlib
from termcolor import colored, cprint

ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
sys.path.append(ROOT_DIR)

from multiprocessing.managers import SharedMemoryManager
from codebase.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from codebase.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


class Command(enum.Enum):
    STOP = 0
    ACTIVATE = 1


class Robotiq85(mp.Process):
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        frequency: int,
        receive_keys: Optional[List],
        launch_timeout: int = 3,
        soft_real_time: bool = False,
        get_max_k: int = 128,
        verbose: bool = False,
    ):
        super().__init__(name="ROBOTIQ85Controller")

        self.client = None
        for _ in range(20):
            if not self.connectToDevice("/dev/ttyUSB0"):
                print("reconnecting to robotiq85...")
                time.sleep(0.2)
        if not self.client.connect():
            raise Exception("gripper connect error")

        self.reset()
        self.activate(timeout=0.05)

        self.launch_timeout = launch_timeout
        self.frequency = frequency
        self.get_max_k = get_max_k
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        # build input queue
        example = {
            "cmd": Command.ACTIVATE.value,
            "target_pose": np.zeros((1,), dtype=np.float64),
            "duration": 0.0,
        }

        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256,
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = ["OpenOrClose"]

        example = dict()
        for key in receive_keys:
            example[key] = 0  # open
        example["gripper_receive_timestamp"] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        self.ready_event = mp.Event()
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
                f"[Robotiq85Controller] Controller process spawned at {self.pid}",
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

    def execute(self, pose, duration=0.1):
        # assert self.is_alive()
        pose = np.array(pose)
        assert pose.shape == (1,)

        message = {
            "cmd": Command.ACTIVATE.value,
            "target_pose": pose,
            "duration": duration,
        }
        self.input_queue.put(message)

    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def run(self):
        cprint("Now Gripper threading is running!", "yellow")
        if self.soft_real_time:
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))

        try:
            # init pose
            self.reset()
            self.activate(timeout=0.05)

            # main loop
            dt = 1.0 / self.frequency

            iter_idx = 0
            target_pose = 0  # open
            curr_status = "OPEN"
            keep_running = True
            while keep_running:
                t_start = time.perf_counter()

                t_now = time.monotonic()
                # update robot state
                state = dict()
                if np.around(target_pose):  # close only when it's open
                    if curr_status == "OPEN":
                        self.close()
                        curr_status = "CLOSE"
                else:  # open only when it's closed
                    if curr_status == "CLOSE":
                        self.open()
                        curr_status = "OPEN"
                for key in self.receive_keys:
                    state[key] = target_pose
                state["gripper_receive_timestamp"] = time.time()
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
                    elif cmd == Command.ACTIVATE.value:
                        target_pose = command["target_pose"]
                        duration = float(command["duration"])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration

                        if self.verbose:
                            cprint(
                                f"[ROBOTIQ85Controller] New pose target: {target_pose}",
                                "red",
                            )
                    else:
                        keep_running = False
                        break
                # regulate frequency
                t_end = time.perf_counter()
                if t_end - t_start < dt:
                    time.sleep(dt - t_end + t_start)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    print("Gripper ready event is set!!!")
                    self.ready_event.set()
                iter_idx += 1

        finally:
            self.open()
            self.disconnectFromDevice()

    def connectToDevice(self, device):
        """Connection to the client"""
        self.client = ModbusSerialClient(
            method="rtu",
            port=device,
            stopbits=1,
            bytesize=8,
            baudrate=115200,
            timeout=0.2,
        )
        # self.client = ModbusClient('192.168.1.103', 502)
        if not self.client.connect():
            print("Unable to connect to %s" % device)
            return False
        return True

    def disconnectFromDevice(self):
        """Close connection"""
        self.client.close()

    def _send_command(self, data):
        """Send a command to the Gripper - the method takes a list of uint8 as an argument.
        The meaning of each variable depends on the Gripper model
        (see support.robotiq.com for more details)
        """
        # make sure data has an even number of elements
        if len(data) % 2 == 1:
            data.append(0)

        # Initiate message as an empty list
        message = []

        # Fill message by combining two bytes in one register
        for i in range(int(len(data) / 2)):
            message.append((data[2 * i] << 8) + data[2 * i + 1])

        # To do!: Implement try/except
        self.client.write_registers(0x03E8, message, unit=0x0009)

    def _get_status(self, numBytes):
        """Sends a request to read, wait for the response and returns the Gripper status.
        The method gets the number of bytes to read as an argument"""
        numRegs = int(ceil(numBytes / 2.0))

        # To do!: Implement try/except
        # Get status from the device
        response = self.client.read_holding_registers(0x07D0, numRegs, unit=0x0009)

        # Instantiate output as an empty list
        output = []

        # Fill the output with the bytes in the appropriate order
        for i in range(0, numRegs):
            output.append((response.getRegister(i) & 0xFF00) >> 8)
            output.append(response.getRegister(i) & 0x00FF)

        # Output the result
        return output

    def getStatus(self):
        """Request the status from the gripper and return it in the Robotiq2FGripper_robot_input msg type."""
        # Acquire status from the Gripper
        status = self._get_status(6)
        # Message to output
        message = {}
        # 夹爪是否正常工作
        message["gACT"] = (status[0] >> 0) & 0x01
        # 夹爪是否正在移动,移动时不能接收下一个指令
        message["gGTO"] = (status[0] >> 3) & 0x01
        message["gSTA"] = (status[0] >> 4) & 0x03
        # 是否检查到物体，0x00正在运动中没有物体，0x01物体在外面，0x02物体在里面，0x03运动到指定位置没有物体
        message["gOBJ"] = (status[0] >> 6) & 0x03
        # 错误信息
        message["gFLT"] = status[2]
        # 需要达到的位置，0x00全开，0xFF全闭
        message["gPR"] = status[3]
        # 当前位置
        message["gPO"] = status[4]
        # 10×电流=gCu(mA)
        message["gCU"] = status[5]
        return message

    def sendCommand(self, command):
        """把字典转化为可以直接发送的数组，并发送"""
        # 限制数值上下限
        for n in "rACT rGTO rATR".split():
            command[n] = int(np.clip(command.get(n, 0), 0, 1))
        for n in "rPR rSP rFR".split():
            command[n] = int(np.clip(command.get(n, 0), 0, 255))
        # 转换为要发送的数组
        message = []
        message.append(
            command["rACT"] + (command["rGTO"] << 3) + (command["rATR"] << 4)
        )
        message.append(0)
        message.append(0)
        for n in "rPR rSP rFR".split():
            message.append(command[n])
        return self._send_command(message)

    def _is_ready(self, status=None):
        status = status or self.getStatus()
        return status["gSTA"] == 3 and status["gACT"] == 1

    def is_reset(self, status=None):
        status = status or self.getStatus()
        return status["gSTA"] == 0 or status["gACT"] == 0

    def is_moving(self, status=None):
        status = status or self.getStatus()
        return status["gGTO"] == 1 and status["gOBJ"] == 0

    def is_stopped(self, status=None):
        status = status or self.getStatus()
        return status["gOBJ"] != 0

    def object_detected(self, status=None):
        status = status or self.getStatus()
        return status["gOBJ"] == 1 or status["gOBJ"] == 2

    def get_fault_status(self, status=None):
        status = status or self.getStatus()
        return status["gFLT"]

    def get_pos(self, status=None):
        status = status or self.getStatus()
        po = status["gPO"]
        # TODO:这里的变换还要调一下
        return np.clip(0.14 / (13.0 - 230.0) * (po - 230.0), 0, 0.14)

    def get_req_pos(self, status=None):
        status = status or self.getStatus()
        pr = status["gPR"]
        # TODO:这里的变换还要调一下
        return np.clip(0.14 / (13.0 - 230.0) * (pr - 230.0), 0, 0.14)

    def is_closed(self, status=None):
        status = status or self.getStatus()
        return status["gPO"] >= 230

    def is_opened(self, status=None):
        status = status or self.getStatus()
        return status["gPO"] <= 13

    def get_current(self, status=None):
        status = status or self.getStatus()
        return status["gCU"] * 0.1

    def wait_until_stopped(self, timeout=None):
        start_time = time.time()
        while not self.is_reset() and (
            not timeout or (time.time() - start_time) < timeout
        ):
            if self.is_stopped():
                return True
            time.sleep(0.1)
        return False

    def wait_until_moving(self, timeout=None):
        start_time = time.time()
        while not self.is_reset() and (
            not timeout or (time.time() - start_time) < timeout
        ):
            if not self.is_stopped():
                return True
            time.sleep(0.1)
        return False

    def reset(self):
        cmd = {n: 0 for n in "rACT rGTO rATR rPR rSP rFR".split()}
        self.sendCommand(cmd)

    def activate(self, timeout=None):
        cmd = dict(rACT=1, rGTO=1, rATR=0, rPR=0, rSP=255, rFR=150)
        self.sendCommand(cmd)
        start_time = time.time()
        while not timeout or (time.time() - start_time) < timeout:
            if self._is_ready():
                return True
            time.sleep(0.1)
        return False

    def auto_release(self):
        cmd = {n: 0 for n in "rACT rGTO rATR rPR rSP rFR".split()}
        cmd["rACT"] = 1
        cmd["rATR"] = 1
        self.sendCommand(cmd)

    def goto(self, pos, vel=0.1, force=50, block=False, timeout=None):
        cmd = {n: 0 for n in "rACT rGTO rATR rPR rSP rFR".split()}
        cmd["rACT"] = 1
        cmd["rGTO"] = 1
        # cmd['rPR'] = int(np.clip((13.-230.)/0.14 * pos + 230., 0, 255))
        cmd["rPR"] = int(np.clip((0.0 - 230.0) / 0.085 * pos + 230.0, 0, 255))
        cmd["rSP"] = int(np.clip(255.0 / (0.1 - 0.013) * (vel - 0.013), 0, 255))
        cmd["rFR"] = int(np.clip(255.0 / (100.0 - 30.0) * (force - 30.0), 0, 255))
        self.sendCommand(cmd)
        time.sleep(0.1)
        if block:
            if not self.wait_until_moving(timeout):
                return False
            return self.wait_until_stopped(timeout)
        return True

    def _stop(self, block=False, timeout=-1):
        cmd = {n: 0 for n in "rACT rGTO rATR rPR rSP rFR".split()}
        cmd["rACT"] = 1
        cmd["rGTO"] = 0
        self.sendCommand(cmd)
        time.sleep(0.1)
        if block:
            return self.wait_until_stopped(timeout)
        return True

    def open(self, vel=2, force=100, block=False, timeout=-1):
        if self.is_opened():
            return True
        return self.goto(1.0, vel, force, block=block, timeout=timeout)

    def close(self, vel=1.0, force=10, block=False, timeout=-1):
        if self.is_closed():
            return True
        return self.goto(-1.0, vel, force, block=block, timeout=timeout)


