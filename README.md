# KUKA-Controller

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
$ python test/simTest.py
```

## Real Environment Details
- KUKA LBR iiwa 7 R800; 
- Robotiq85;
- SpaceMouse Wireless;

<div align="center">
    <img src="assets/real_env.jpg" width="50%" />
</div>

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

In order to run the data collection code, execute the example command below:
```shell
python demo_real_robot.py -o "data/test_data" --robot_ip "172.31.1.147" --robot_port 30001
```

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

**How to obtain target_pose/action?**

```python
# current EEF pose (without any transformation)
target_pose

# spacemouse states
sm_state = sm.get_motion_state_transformed()
## scale spacemouse states
dpos = (
    sm_state[:3] * (env.max_pos_speed / frequency) * pos_sensitivity
)
drot_xyz = (
    sm_state[3:]
    * (env.max_rot_speed / frequency)
    * np.array([1, 1, -1])
    * rot_sensitivity
)
## spacemouse euler -> quat
drot = st.Rotation.from_euler("xyz", drot_xyz)

# target EEF pos = current EEF pos + dpos
target_pose[:3] += dpos
# target EEF rot = drot * current EEF rot quat
target_pose[3:] = (
    drot * st.Rotation.from_euler("zyx", target_pose[3:])
).as_euler("zyx")
```

The final target_pose is our next step action, and then we send this command to instruct remoter to move, and final reply an actual EEF pose to client. Below is a example of target pose and difference between actual EEF pose and target pose.

```text
[614.42  -2.24 318.24   3.14  -0.     3.14]  |  [-0.03  0.24  0.55 -0.   -0.    0.  ]
[614.42  -1.88 319.11   3.14  -0.     3.14]  |  [-0.05  0.26  0.56 -0.   -0.    0.  ]
[614.42  -1.5  319.98   3.14  -0.     3.14]  |  [-0.03  0.26  0.68  0.    0.    0.  ]
[614.42  -1.12 320.8    3.14  -0.     3.14]  |  [ 0.01  0.26  0.67 -0.    0.    0.  ]
[614.42  -0.74 321.56   3.14  -0.     3.14]  |  [ 0.02  0.27  0.51 -0.    0.    0.  ]
[614.42  -0.38 322.31   3.14  -0.     3.14]  |  [ 0.02  0.27  0.46  0.   -0.    0.  ]
[614.42  -0.05 322.92   3.14  -0.     3.14]  |  [ 0.02  0.2   0.42 -0.    0.    0.  ]
[614.42   0.19 323.39   3.14  -0.     3.14]  |  [-0.04  0.17  0.32 -0.   -0.    0.  ]
[614.42   0.34 323.67   3.14  -0.     3.14]  |  [-0.06  0.18  0.25 -0.   -0.    0.  ]
```

### Model Training

Add some supplement files in diffusion_policy. (Before doing this, make sure you've already read `README.md` of <a ref="https://github.com/real-stanford/diffusion_policy">diffusion_policy</a>. )
```shell
[KUKA-Controller] $ cp codebase/sup_files/dataset/real_lift_image_dataset.py codebase/diffusion_policy/diffusion_policy/dataset
[KUKA-Controller] $ cp codebase/sup_files/config/task/real_pusht_image.yaml codebase/diffusion_policy/diffusion_policy/dataset/config/task
```

The modify the config files including specific task name and dataset path. For example, if you decide to train your model with `train_diffusion_transformer_real_hybrid_workspace.yaml`, first you need to replace `task: real_pusht_image` with `task: real_pusht_image`, then replace `dataset_path` with your desired path in `real_pusht_image.yaml`.

Just use Diffusion Policy to train that model with our collected data.
```shell
[KUKA-Controller/codebase/diffusion_policy] $ python train.py --config-dir=. --config-name=<ws_cfg_file>
```

### Evaluate Real Robot
The inference loop can be demomstrated below:
```python
# load chechpoint
# checkpoint to policy
# Policy control loop
## get observations
## run inference
## convert policy action to env actions 
## clip action
## execute actions
```
```shell
python eval_real_robot.py --input_path <ckpt_path> --output_path <opt_dir>
```

**Synchronize the Speend of Remoter and Client**

Assume that we have below in remoter:

$f_c$: communication frequency

$pVel_{max}$: real robot position velocity (mm/s)

$rVel_{max}$: real robot rotation velocity (rad/s)

Assume that we have below in client,

$opt_{sm}$: SpaceMouse output

$f_r$: real env frequency

$p_s$: delta position sensitivity [0.0, 1.0], the less it is, the smoother remoter will be but slower.

$p_d=pVel_{max}/f_r$: desired position speed 

$delta_p=opt_{sm}[:3]\times p_d\times p_s$: position action

$r_s$: delta rotation sensitivity [0.0, 1.0], the less it is, the smoother remoter will be but slower.

$r_d=rVel_{max}/f_r$: desired rotation speed 

$delta_r=opt_{sm}[3:]\times r_d\times r_s$: rotation action



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
- [x] Implement keyboard control.
- [x] Robot, Gripper, data collector, every of these parts need to hybrid from threading.
