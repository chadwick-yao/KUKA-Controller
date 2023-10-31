import numpy as np
import utils.transform_utils as T
import copy
from codebase.real_world.interpolators.base_interpolator import Interpolator
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


class LinearInterpolator(Interpolator):
    def __init__(
        self,
        ndim: int,
        controller_freq: int,
        policy_freq: int,
        ramp_ratio: float = 0.2,
        use_delta_goal: bool = False,
        ori_interpolate: str = None,
    ) -> None:
        super().__init__()

        self.dim = ndim  # Number of dimensions to interpolate
        self.times = [0, 1]
        self.ori_interpolate = ori_interpolate  # interpolating orientation or not
        self.step = 0  # current step of the interpolator
        self.total_steps = np.ceil(
            ramp_ratio * controller_freq / policy_freq
        )  # total num steps per interpolator action
        self.inter_times = np.linspace(0, 1, int(self.total_steps))
        self.use_delta_goal = use_delta_goal  # whether to use delta or absolute goals
        self.set_states(dim=ndim, ori=ori_interpolate)

    def set_states(self, dim=None, ori=None):
        self.dim = dim if dim is not None else self.dim
        self.ori_interpolate = ori if ori is not None else self.ori_interpolate

        if self.ori_interpolate is not None:
            if self.ori_interpolate == "euler":
                self.start = np.zeros(3)
            elif self.ori_interpolate == "quat":
                self.start = np.array([0, 0, 0, 1])
            else:
                raise NotImplementedError
        else:
            self.start = np.zeros(self.dim)
        self.goal = copy.deepcopy(self.start)

    def set_start(self, start):
        self.start = copy.deepcopy(start)
        self.goal = copy.deepcopy(start)
        self.interp_rots = None
        self.step = 0

    def set_goal(self, goal):
        assert len(goal) == self.dim, f"Goal dimension should be {self.dim}."

        self.start = copy.deepcopy(self.goal)
        self.goal = goal
        self.step = 0

    def get_interpolated_goal(self):
        super().get_interpolated_goal()

        x = copy.deepcopy(self.start)

        if self.ori_interpolate is not None:
            goal = copy.deepcopy(self.goal)
            if self.ori_interpolate == "euler":
                key_rots = R.from_euler("xyz", [x, goal], degrees=False)
            elif self.ori_interpolate == "quat":
                key_rots = R.from_quat([x, goal])

            if self.step == 0:
                slerp = Slerp(self.times, key_rots)
                self.interp_rots = slerp(self.inter_times)

            if self.ori_interpolate == "euler":
                x_current = self.interp_rots.as_euler("xyz", degrees=False)[self.step]
            elif self.ori_interpolate == "quat":
                x_current = self.interp_rots.as_quat()[self.step]
        else:
            dx = (self.goal - x) / (self.total_steps - self.step)
            x_current = x + dx

        if self.step < self.total_steps:
            self.step += 1

        return x_current
