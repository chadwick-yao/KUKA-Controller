from typing import Union
import numbers
import numpy as np
import scipy.interpolate as si


def rotation_distance(a, b) -> float:
    rot_dist = np.linalg.norm(b - a)
    return rot_dist


class JposTrajectoryInterpolator:
    def __init__(self, times: np.ndarray, jposes: np.ndarray):
        assert len(times) >= 1
        assert len(jposes) == len(times)

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
        self,
        jpos,
        time,
        curr_time,
        max_rot_speed=np.inf,
    ) -> "JposTrajectoryInterpolator":
        assert max_rot_speed > 0
        time = max(time, curr_time)

        curr_jpos = self(curr_time)
        rot_dist = rotation_distance(curr_jpos, jpos)
        rot_min_duration = rot_dist / max_rot_speed
        duration = time - curr_time
        duration = max(duration, rot_min_duration)
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        # insert new jpos
        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        jposes = np.append(trimmed_interp.jposes, [jpos], axis=0)

        # create new interpolator
        final_interp = JposTrajectoryInterpolator(times, jposes)
        return final_interp

    def schedule_waypoint(
        self,
        jpos,
        time,
        max_rot_speed=np.inf,
        curr_time=None,
        last_waypoint_time=None,
    ) -> "JposTrajectoryInterpolator":
        assert max_rot_speed > 0
        if last_waypoint_time is not None:
            assert curr_time is not None

        # trim current interpolator to between curr_time and last_waypoint_time
        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                # if insert time is earlier than current time
                # no effect should be done to the interpolator
                return self
            # now, curr_time < time
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                # if last_waypoint_time is earlier than start_time
                # use start_time
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        # end time should be the latest of all times except time
        # after this we can assume order (proven by zhenjia, due to the 2 min operations)

        # Constraints:
        # start_time <= end_time <= time (proven by zhenjia)
        # curr_time <= start_time (proven by zhenjia)
        # curr_time <= time (proven by zhenjia)

        # time can't change
        # last_waypoint_time can't change
        # curr_time can't change
        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)

        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time)
        # after this, all waypoints in trimmed_interp is within start_time and end_time
        # and is earlier than time

        # determine speed
        duration = time - end_time
        end_jpos = trimmed_interp(end_time)
        rot_dist = rotation_distance(jpos, end_jpos)

        rot_min_duration = rot_dist / max_rot_speed
        duration = max(duration, rot_min_duration)
        assert duration >= 0
        last_waypoint_time = end_time + duration

        # insert new jpos
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

        jpos = np.zeros((len(t), 7))
        if self.single_step:
            jpos[:] = self._jposes[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)

            jpos = np.zeros((len(t), 7))
            jpos = self.jpos_interp(t)

        if is_single:
            jpos = jpos[0]
        return jpos
