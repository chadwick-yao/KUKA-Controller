from typing import Any
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st

from typing import Union
import numbers


def jpos_distance(start_jpos, end_jpos):
    start_jpos = np.array(start_jpos)
    end_jpos = np.array(end_jpos)

    return np.abs(np.max(end_jpos - start_jpos))


class JposTrajectoryInterpolator:
    """
    Joint position interpolator
    """

    def __init__(
        self,
        times: np.ndarray,
        jposes: np.ndarray,
    ) -> None:
        assert len(times) >= 1 and len(jposes) == len(times)

        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(jposes, np.ndarray):
            jposes = np.array(jposes)

        if len(times) == 1:
            # special treatment for single step interpolation
            self.single_step = True
            self._times = times
            self._jposes = jposes
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])

            # use linear interpolator
            self.jpos_interp = si.interp1d(times, jposes, axis=0, assume_sorted=True)

    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        else:
            return self.jpos_interp.x

    @property
    def jposes(self) -> np.ndarray:
        if self.single_step:
            return self._jposes
        else:
            return self.jpos_interp.y

    def trim(self, start_t: float, end_t: float) -> "JposTrajectoryInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        # remove duplicates, Slerp requires strictly increasing x
        all_times = np.unique(all_times)

        # interpolate
        all_jposes = self(all_times)

        return JposTrajectoryInterpolator(times=all_times, jposes=all_jposes)

    def drive_to_waypoint(
        self, jpos, time, curr_time, max_rot_speed=np.inf
    ) -> "JposTrajectoryInterpolator":
        assert max_rot_speed > 0
        time = max(time, curr_time)

        curr_jpos = self(curr_time)
        jpos_dist = jpos_distance(jpos, curr_jpos)

        jpos_min_duration = jpos_dist / max_rot_speed

        duration = time - curr_time
        duration = max(duration, jpos_min_duration)
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        # insert new pose
        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        jposes = np.append(trimmed_interp.jposes, [jpos], axis=0)

        # create new interpolator
        final_interp = JposTrajectoryInterpolator(times, jposes)
        return final_interp

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])

        if self.single_step:
            jpos = self._jposes[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)

            jpos = self.jpos_interp(t)

        if is_single:
            jpos = jpos[0]

        return jpos
