# Coppeliasim-Python

Coppeliasim-Python provides users an easy tool to manipulate robots in CoppeliaSim with SpaceMouse (3D Mouse). 

<div align="center">
    <img src="assets/demo.gif" style="width: 200px;"  />
</div>


## Usage

### SpaceMouse

#### Setup

- hidapi python package. `pip install hidapi`
- udev system: `sudo apt-get install libhidapi-dev`

**udev rules**: In order for hidapi to open the device without sudo, we need to do the following steps. First of all, create a rule file xx-spacemouse.rules under the folder /etc/udev/rules.d/ (Replace xx with a number larger than 50).

```bash
sudo touch /etc/udev/rules.d/77-spacemouse.rules

echo "KERNEL==\"hidraw*\", ATTRS{idVendor}==\"256f\", ATTRS{idProduct}==\"c62e\", MODE=\"0666\", GROUP=\"plugdev\"" > /etc/udev/rules.d/77-spacemouse.rules

echo "SUBSYSTEM==\"usb\", ATTRS{idVendor}==\"256f\", ATTRS{idProduct}==\"c62e\", MODE=\"0666\", GROUP=\"plugdev\"" >> /etc/udev/rules.d/77-spacemouse.rules
```

Then we need to reload the defined udev rule to take effect. Reload can be done through:

```bash
$ sudo udevadm control --reload-rules
```
Get vendor_id and product_id of your Spacemouse.

```python
import hid
hid.enumerate()
```
The python code above will print all HID information, just locate the information about SpaceMouse.

#### Test Connection

```bash
$ python common/spacemouse.py
```
### CoppeliaSim

- Open `example/iiwa7.ttt` with CoppeliaSim.
- Start simulation.

### SpaceMouse to Manipulate Robots in SimEnv

```bash
$ python simTest.py
```

## Real Environment Details

### Threading Settings
- real_env (main threading): Get observations, send commands to child threading, and recording videos, actions and observations.
    - iiwaPy3 (Child threading): Execute commands, and update shared memory.
    - robotiq85 (Child threading): Execute commands, and update shared memory.
    - multirealsense (Multi child threading): Execute commands, and update shared memory.
- spacemouse (main threading): Update shared memory.
- keyboard (main threading): Update stage (stage means how many parts that the whole episode can be separated.)

### Data Collection
In `demo_real_robot.py`, `c` refers to starting recording a new episode, `s` refers to ending one episode, and `q` means quitting the process. There uses data from spacemouse as delta pose and regards the transformed pose as the next step action, then the next step action will call `execute_actions` in `real_env.py` to control child threading.

Function `get_obs` from `real_env.py` will also recall its child threading function to get observations, and finally save them in `Dict`. After ending one episode, the `Dict` will be saved in the shared memory.

The saved data form is like below:

>- data
>   - action
>   - timestamps
>   - stage
>   - eef_rot
>   - eef_pos
>   - joints_pos
>- meta
>   - episode_ends
>- videos

### Model Training

Just use Diffusion Policy to train that model with our collected data.

### Evaluate Real Robot
```python
# load chechpoint
# checkpoint to policy
# Policy control loop
## get observations
## run inference
## convert policy action to env actions (Here I modified because I don't know the original code is doing)
if delta_action:
    assert len(action) == 1
    if perv_target_pose is None:
        perv_target_pose = obs['robot_eef_pose'][-1]
    this_target_pose = perv_target_pose.copy()
    this_target_pose[[0,1]] += action[-1]
    perv_target_pose = this_target_pose
    this_target_poses = np.expand_dims(this_target_pose, axis=0)
else:
    this_target_poses = np.zeros((len(action), len(target_pose)), dtype=np.float64)
    this_target_poses[:] = target_pose
    this_target_poses[:,[0,1]] = action
## deal with timing
action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
    ) * dt + obs_timestamps[-1]
action_exec_latency = 0.01
curr_time = time.time()
is_new = action_timestamps > (curr_time + action_exec_latency)
if np.sum(is_new) == 0:
    # exceeded time budget, still do something
    this_target_poses = this_target_poses[[-1]]
    # schedule on next available step
    next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
    action_timestamp = eval_t_start + (next_step_idx) * dt
    print('Over budget', action_timestamp - curr_time)
    action_timestamps = np.array([action_timestamp])
else:
    this_target_poses = this_target_poses[is_new]
    action_timestamps = action_timestamps[is_new]

## clip actions (I commended)
this_target_poses[:,:2] = np.clip(
    this_target_poses[:,:2], [0.25, -0.45], [0.77, 0.40])

## execute actions
```


## TO DO
- [x] Implement base robot controlling class.
- [x] Implement SpaceMouse API to control iiwa.
    - [x] thread to listen SpaceMouse
    - [x] Dof controlling
    - [x] Button controlling, i.e restart or grip
        - [x] callback
        - [x] grip controlling
        - [x] reset
- [x] Real world. (Real machine control)
    - [x] SpaceMouse
    - [x] RealSense Camera
    - [x] Gripper
    - [x] Data Collection

NOT IMPORTANT:
- [ ] Implement keyboard control.
- [x] Robot, Gripper, data collector, every of these parts need to hybrid from threading.