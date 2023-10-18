import api.sim as sim
from abc import ABCMeta, abstractMethod


class BaseRobot(metaclass=ABCMeta):
    """ base robot class """

    def __init__(self) -> None:
        pass