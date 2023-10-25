<div align="center">
    <img src="assets/demo.gif" style="width: 200px;"  />
</div>

# Coppeliasim-Python

## Usage

### SpaceMouse

#### Setup

- hidapi python package. `pip install hidapi`
- udev system: `sudo apt-get install libhidapi-dev`

**udev rules**: In order for hidapi to open the device without sudo, we need to do the following steps. First of all, create a rule file xx-spacemouse.rules under the folder /etc/udev/rules.d/ (Replace xx with a number larger than 50).

```bash
KERNEL=="hidraw*", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c62e", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c62e", MODE="0666", GROUP="plugdev"
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
$ pthon simTest.py
```

## TO DO
- [x] Implement base robot controlling class.
- [x] Implement SpaceMouse API to control iiwa.
    - [x] thread to listen SpaceMouse
    - [x] Dof contralling
    - [ ] Button controlling, i.e restart or grip
        - [x] callback
        - [x] grip controlling
        - [x] reset
- [ ] Data Collection Base class
- [ ] Real world. (Real machine contral)

NOT IMPORTANT:
- [ ] Implement keyboard control.