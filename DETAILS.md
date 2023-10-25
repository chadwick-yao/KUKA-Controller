# SpaceMouse控制CoppeliaSim仿真环境的机器人

## 控制回路
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
    robot.setup_all()
    robot.start_control()
    while True:
        time.sleep(0.01)
        robot.input2action()
```

在连接仿真机器人后，会通过`input2action`函数执行来自于3D鼠标的指令（position + rotation），伪代码如下：

```python
# 1. obtain current status of SpaceMouse
state: dict = self.get_controller_state

dpos, rotation, raw_rotation, grasp, reset = [
    state[key]
    for key in state.keys()
]
# 2. current pose + action from SpaceMouse = next timestep pose
action = (dpos, raw_rotation)
orig_pose = self._get_pose(self.targetHanle, use_quat=False)
target_pose = (action[0] + orig_pose[0], action[1] + orig_pose[1])
self._set_pose(self.targetHanle, target_pose)
```

接着会处理按钮指令，如下，执行结束之后会delay一段时间，这是为了解决通信以及仿真延迟问题：

```python
# gripper position setting
grasp = 1 if self.single_click_and_hold else -1
res, retInts, retFloats, retStrings, retBuffer = sim.simxCallScriptFunction(self.clientID, "RG2",
                                                sim.sim_scripttype_childscript,'rg2_OpenClose',[grasp],[],[],b'',sim.simx_opmode_blocking)
```

当把信息传递到仿真环境并set好EEF的位置和角度后，仿真环境会进行IK调整机械臂的关节位置，具体代码如下（初始化函数，actuate函数以及cleanup函数）：

```lua
function sysCall_init()
    -- Build a kinematic chain and 2 IK groups (undamped and damped) inside of the IK plugin environment,
    -- based on the kinematics of the robot in the scene:
    -- There is a simple way, and a more elaborate way (but which gives you more options/flexibility):

    -- Simple way:
    local simBase=sim.getObject('.')
    local simTip=sim.getObject('./tip')
    local simTarget=sim.getObject('./target')

    ikEnv=simIK.createEnvironment() -- create an IK environment
    ikGroup_undamped=simIK.createGroup(ikEnv) -- create an IK group
    simIK.setGroupCalculation(ikEnv,ikGroup_undamped,simIK.method_pseudo_inverse,0,6) -- set its resolution method to undamped
    simIK.addElementFromScene(ikEnv,ikGroup_undamped,simBase,simTip,simTarget,simIK.constraint_pose) -- create an IK element based on the scene content
    ikGroup_damped=simIK.createGroup(ikEnv) -- create another IK group
    simIK.setGroupCalculation(ikEnv,ikGroup_damped,simIK.method_damped_least_squares,1,99) -- set its resolution method to damped
    simIK.addElementFromScene(ikEnv,ikGroup_damped,simBase,simTip,simTarget,simIK.constraint_pose) -- create an IK element based on the scene content
    
    -- Elaborate way:
    --[[
    simBase=sim.getObject('.')
    simTip=sim.getObject('./tip')
    simTarget=sim.getObject('./target')
    simJoints={}
    for i=1,7,1 do
        simJoints[i]=sim.getObject('./joint',{index=i-1})
    end
    ikJoints={}

    ikEnv=simIK.createEnvironment() -- create an IK environment
    ikBase=simIK.createDummy(ikEnv) -- create a dummy in the IK environemnt
    simIK.setObjectMatrix(ikEnv,ikBase,-1,sim.getObjectMatrix(simBase,sim.handle_world)) -- set that dummy into the same pose as its CoppeliaSim counterpart
    local parent=ikBase
    for i=1,#simJoints,1 do -- loop through all joints
        ikJoints[i]=simIK.createJoint(ikEnv,simIK.jointtype_revolute) -- create a joint in the IK environment
        simIK.setJointMode(ikEnv,ikJoints[i],simIK.jointmode_ik) -- set it into IK mode
        local cyclic,interv=sim.getJointInterval(simJoints[i])
        simIK.setJointInterval(ikEnv,ikJoints[i],cyclic,interv) -- set the same joint limits as its CoppeliaSim counterpart joint
        simIK.setJointPosition(ikEnv,ikJoints[i],sim.getJointPosition(simJoints[i])) -- set the same joint position as its CoppeliaSim counterpart joint
        simIK.setObjectMatrix(ikEnv,ikJoints[i],-1,sim.getObjectMatrix(simJoints[i],sim.handle_world)) -- set the same object pose as its CoppeliaSim counterpart joint
        simIK.setObjectParent(ikEnv,ikJoints[i],parent,true) -- set its corresponding parent
        parent=ikJoints[i]
    end
    ikTip=simIK.createDummy(ikEnv) -- create the tip dummy in the IK environment
    simIK.setObjectMatrix(ikEnv,ikTip,-1,sim.getObjectMatrix(simTip,sim.handle_world)) -- set that dummy into the same pose as its CoppeliaSim counterpart
    simIK.setObjectParent(ikEnv,ikTip,parent,true) -- attach it to the kinematic chain
    ikTarget=simIK.createDummy(ikEnv) -- create the target dummy in the IK environment
    simIK.setObjectMatrix(ikEnv,ikTarget,-1,sim.getObjectMatrix(simTarget,sim.handle_world)) -- set that dummy into the same pose as its CoppeliaSim counterpart
    simIK.setTargetDummy(ikEnv,ikTip,ikTarget) -- link the two dummies
    ikGroup_undamped=simIK.createGroup(ikEnv) -- create an IK group
    simIK.setGroupCalculation(ikEnv,ikGroup_undamped,simIK.method_pseudo_inverse,0,6) -- set its resolution method to undamped
    simIK.setGroupFlags(ikEnv,ikGroup_undamped,1+2+4+8) -- make sure the robot doesn't shake if the target position/orientation wasn't reached
    local ikElementHandle=simIK.addElement(ikEnv,ikGroup_undamped,ikTip) -- add an IK element to that IK group
    simIK.setElementBase(ikEnv,ikGroup_undamped,ikElementHandle,ikBase) -- specify the base of that IK element
    simIK.setElementConstraints(ikEnv,ikGroup_undamped,ikElementHandle,simIK.constraint_pose) -- specify the constraints of that IK element
    ikGroup_damped=simIK.createGroup(ikEnv) -- create another IK group
    simIK.setGroupCalculation(ikEnv,ikGroup_damped,simIK.method_damped_least_squares,1,99) -- set its resolution method to damped
    local ikElementHandle=simIK.addElement(ikEnv,ikGroup_damped,ikTip) -- add an IK element to that IK group
    simIK.setElementBase(ikEnv,ikGroup_damped,ikElementHandle,ikBase) -- specify the base of that IK element
    simIK.setElementConstraints(ikEnv,ikGroup_damped,ikElementHandle,simIK.constraint_pose) -- specify the constraints of that IK element
    --]]
end

function sysCall_actuation()
    -- There is a simple way, and a more elaborate way (but which gives you more options/flexibility):
    
    -- Simple way:
    if simIK.handleGroup(ikEnv,ikGroup_undamped,{syncWorlds=true})~=simIK.result_success then -- try to solve with the undamped method
        -- the position/orientation could not be reached.
        simIK.handleGroup(ikEnv,ikGroup_damped,{syncWorlds=true,allowError=true}) -- try to solve with the damped method
    end
    
    -- Elaborate way:
    --[[
    simIK.setObjectMatrix(ikEnv,ikTarget,ikBase,sim.getObjectMatrix(simTarget,simBase)) -- reflect the pose of the target dummy to its counterpart in the IK environment

    if simIK.handleGroup(ikEnv,ikGroup_undamped)~=simIK.result_success then -- try to solve with the undamped method
        -- the position/orientation could not be reached.
        simIK.handleGroup(ikEnv,ikGroup_damped) -- try to solve with the damped method
    end
    
    for i=1,#simJoints,1 do
        sim.setJointPosition(simJoints[i],simIK.getJointPosition(ikEnv,ikJoints[i])) -- apply the joint values computed in the IK environment to their CoppeliaSim joint counterparts
    end
    --]]
end 

function sysCall_cleanup() 
    simIK.eraseEnvironment(ikEnv) -- erase the IK environment
end 

```



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
