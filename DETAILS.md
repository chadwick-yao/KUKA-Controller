# SpaceMouse控制CoppeliaSim仿真环境的机器人

## 控制回路

## 连接到HID (SpaceMouse) 

### 定义设备信息
首先需要对每一款设备定义好设备信息，包括HID ID，product ID以及按钮mapping的信息，这里以SpaceMouse Wireless为例：
```python
"SpaceMouse Wireless": DeviceSpec(
    name="SpaceMouse Wireless",
    # vendor ID and product ID
    hid_id=[0x256F, 0xC62E],
    # LED HID usage code pair
    led_id=[0x8, 0x4B],
    mappings={
        "x": AxisSpec(channel=1, byte1=1, byte2=2, scale=1),
        "y": AxisSpec(channel=1, byte1=3, byte2=4, scale=-1),
        "z": AxisSpec(channel=1, byte1=5, byte2=6, scale=-1),
        "pitch": AxisSpec(channel=1, byte1=7, byte2=8, scale=-1),
        "roll": AxisSpec(channel=1, byte1=9, byte2=10, scale=-1),
        "yaw": AxisSpec(channel=1, byte1=11, byte2=12, scale=1),
    },
    button_mapping=[
        ButtonSpec(channel=3, byte=1, bit=0),  # LEFT
        ButtonSpec(channel=3, byte=1, bit=1),  # RIGHT
    ],  # FIT
    axis_scale=350.0,
)
```
参考了设备信息，另外定义并支持SpaceNavigator，SpaceMouse USB，SpaceMouse Compact，SpacePilot等系类设备。

### 实现3D鼠标基类
这个抽象类主要功能有以下四者，通过给定的ID进行设备的打开/关闭，读取设备传给PC的byte数据，以及最后的处理字节数据和对应按键的触发函数的process函数。
```python
class DeviceSpec(object):
    # ...
    def open(self):
        pass
    # ...
    def close(self):
        pass
    # ...
    def read(self):
        pass
    # ...
    def process(self):
        pass
```
process函数中实现了Dof触发函数接口和按钮触发函数接口，表示如下：
```python
class Config:
    """ Create new config file with correct structure and check that the configuration has correct parts """
    
    def __init__(self,
                 callback: Callable[[object], None] = None,
                 dof_callback: Callable[[object], None] = None,
                 dof_callback_arr: List[DofCallback] = None,
                 button_callback: Callable[[object, list], None] = None,
                 button_callback_arr: List[ButtonCallback] = None) -> None:
        check_config(callback, dof_callback, dof_callback_arr, button_callback, button_callback_arr)
        self.callback = callback
        self.dof_callback = dof_callback
        self.dof_callback_arr = dof_callback_arr
        self.button_callback = button_callback
        self.button_callback_arr = button_callback_arr
```
callback是所有类型按键的回调函数，dof_callback和button_callback是所有一类的按键均会触发的回调函数，dof_callback_arr和button_callback_arr是各类按键内对应按钮的触发回调函数。

## 远程操控机器人

### IK Mode
在仿真环境中，把写好的IK lua代码覆盖机械臂初始代码。

### Base Robot

考虑到代码的可扩展性，计划是希望能够控制CoppelliaSim中所有robot，不限于不可移动的机械臂，也包括各类可移动的robot。所以首先抽象出一个BaseRobot的基类描述所有仿真机器人，它的功能如下（之后再考虑维护和更新）：
```python
class BaseRobot(metaclass=ABCMeta):
    def __init__(self,
                 RobotName: str,
                 TargetName: str,
                 DataDir: str,
                 DefaultCam: Union[List, str, None] = None,
                 OtherCam: Union[List, str, None] = None,
                 Address: str = "127.0.0.1",
                 Port: int = 19999,
                 ) -> None:
        pass
    # ...
    @abstractmethod
    def run(self):
        pass
    @abstractmethod
    def input2action(self):
        pass
    @abstractmethod
    def _setup_robot(self):
        pass
    def _get_pose(self, obj_handle, use_quat=True):
        pass
    def _set_pose(self, obj_handle, target_pose):
        pass
    def _check_pose(self, target_obj_handle, mode="INFO"):
        pass
    def _setup_cameras(self):
        pass
    def _get_camera_data(self, cam_info, need_depth=False):
        pass
```
还有data collection等函数之后实现。

### Manipulator Robot
实现ManipulatorRobot类对机械臂的控制（iiwa, UR系列等）。实现分两部分，一部分使用子线程通过run函数对SpaceMouse监听，主线程用来和仿真远程通信控制机械臂。
```python
def run(self):
    super().run()
    """ Listener method that keeps pulling new message. """
    while True:
        if self._enable:
        ## Read (pos, orient) from SpaceMouse
            _, dof_changed, button_changed = self.HIDevice.read()

            # button function
            if button_changed:
                if self.control_gripper[0] == 0:    # release left button
                    self.single_click_and_hold = False
                elif self.control_gripper[0] == 1:  # press left button
                    self.single_click_and_hold = True
                elif self.control_gripper[1] == 1:  # press right button
                    self._reset_state = 1
                    self._enabled = False
                    self._reset_internal_state()

def input2action(self):
    state: dict = self.get_controller_state

    dpos, rotation, raw_rotation, grasp, reset = [
        state[key]
        for key in state.keys()
    ]

    # if we are resetting, directly return None value
    if reset:
        return None, None
    
    # some pre=processing FIXME

    action = (dpos, raw_rotation)
    orig_pose = self._get_pose(self.targetHanle, use_quat=False)
    target_pose = (action[0] + orig_pose[0], action[1] + orig_pose[1])
    self._set_pose(self.targetHanle, target_pose)

    # gripper position setting
    grasp = 1 if self.single_click_and_hold else -1
    res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(self.clientID, "RG2",
                                                    sim.sim_scripttype_childscript,'rg2_OpenClose',[grasp],[],[],b'',sim.simx_opmode_blocking)
```
后续会继续写mobile robots类，以及除了IK控制以外的控制方式。

远程控制循环：
```python
if __name__=="__main__":
    SpaceMouseConf = DeviceConfig(
        # dof_callback = show_control_state
    )
    robot = ManipulatorRobot(
        SpaceMouseConf,
        Address = "127.0.0.1",
        Port = 19999,
        RobotName = "LBR_iiwa_7_R800",
        TargetName = "targetSphere",
        DataDir = "data",
        ObjName = ["RG2"]
    )
    robot.start_control()
    robot.setup_all()
    robot._reset_internal_state()
    while True:
        time.sleep(0.05)
        robot.input2action()
```
