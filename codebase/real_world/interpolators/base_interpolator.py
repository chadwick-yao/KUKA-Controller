import abc


class Interpolator(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_interpolated_goal(self):
        """
        Provide the next step in interpolation given the remaining steps.
        Returns:
            np.array: Next interpolated step
        """
        pass
