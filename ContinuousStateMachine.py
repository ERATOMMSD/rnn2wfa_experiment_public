import numpy as np
import abc


class ContinuousStateMachine(abc.ABC):
    alphabet: str

    @abc.abstractmethod
    def get_configuration(self, w: str) -> np.ndarray:
        """
        Return the current configuration in np.array of shape (1, -1,)
        :param w:
        :return:
        """
        pass

    @abc.abstractmethod
    def get_value(self, w: str) -> np.float:
        pass

    @abc.abstractmethod
    def get_callings(self) -> int:
        pass
