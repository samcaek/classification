from abc import ABC, abstractmethod


class Investigator(ABC):
    """
    Abstract class uses to regress or classify an example based on its nearest neighbors.
    """

    @abstractmethod
    def investigate(self, nearest_neighbors):
        """
        Investigates an example based on its nearest neighbors.
        :param nearest_neighbors: The nearest neighbors of the example to investigate.
        :return: The predicted output value of the example.
        """
        pass
