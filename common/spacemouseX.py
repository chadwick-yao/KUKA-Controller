from collections.abc import Callable, Iterable, Mapping
from typing import Any
from spnav import (
    spnav_open,
    spnav_poll_event,
    spnav_close,
    SpnavMotionEvent,
    SpnavButtonEvent,
)
from threading import Thread, Event
from collections import defaultdict
import numpy as np
import time


class SpaceMouse(Thread):
    def __init__(self, max_value=500, deadzone=(0, 0, 0, 0, 0, 0), dtype=np.float32):
        """
        Continuously listen to 3D connection space naviagtor events and update states
            max_value: {300, 500} 300 for wired version and 500 for wireless
        """
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()

        super().__init__()

        self.stop_event = Event()
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        print(self.deadzone)
        self.motion_event = SpnavMotionEvent([0, 0, 0], [0, 0, 0], 0)
        self.button_state = defaultdict(lambda: False)
        self.tx_zup_spnav = np.array([[0, 0, -1], [1, 0, 0], [0, 1, 0]], dtype=dtype)

    def get_motion_state(self):
        me = self.motion_event
        state = (
            np.array(me.translation + me.rotation, dtype=self.dtype) / self.max_value
        )
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state

    def get_motion_state_transformed(self):
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_zup_spnav @ state[:3]
        tf_state[3:] = self.tx_zup_spnav @ state[3:]
        return tf_state

    def is_button_pressed(self, button_id):
        return self.button_state[button_id]

    def stop(self):
        self.stop_event.set()
        self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        spnav_open()
        try:
            while not self.stop_event.is_set():
                event = spnav_poll_event()
                if isinstance(event, SpnavMotionEvent):
                    self.motion_event = event
                elif isinstance(event, SpnavButtonEvent):
                    self.button_state[event.bnum] = event.press
                else:
                    time.sleep(1 / 200)
        finally:
            spnav_close()


if __name__ == "__main__":
    low2high = [False, False]
    last_button = [False, False]

    with SpaceMouse(max_value=300) as sm:
        for i in range(1000):
            current_button = [sm.is_button_pressed(0), sm.is_button_pressed(1)]
            print(sm.get_motion_state_transformed()[:3])
            if current_button[0] == True and last_button[0] == False:
                low2high[0] = True
            if current_button[1] == True and last_button[1] == False:
                low2high[1] = True

            if low2high[0]:
                print("Button 1 has been pressed.")
                low2high[0] = False
            if low2high[1]:
                print("Button 2 has been pressed.")
                low2high[1] = False
            last_button = current_button
            time.sleep(1 / 100)
