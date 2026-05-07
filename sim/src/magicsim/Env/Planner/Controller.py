from abc import ABC, abstractmethod

from isaacsim.core.utils.types import ArticulationAction


class BaseController(ABC):
    """[summary]

    Args:
        name (str): [description]
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @abstractmethod
    def forward(self, *args, **kwargs) -> ArticulationAction:
        """A controller should take inputs and returns an ArticulationAction to be then passed to the
           ArticulationController.

        Args:
            observations (dict): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            ArticulationAction: [description]
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets state of the controller."""
        return

    @abstractmethod
    def is_done(self) -> bool:
        """Returns whether the controller has finished its task."""
        return False

    @abstractmethod
    def get_stage(self) -> str:
        """Returns the current stage of the controller."""
        return "default"
