from abc import ABC, abstractmethod

from data_structures.data_set import DataSet


class DataProcessor(ABC):
    """
    Abstract class which processes a DataSet.
    """

    @abstractmethod
    def process(self, data_set: DataSet):
        """
        Processes a DataSet.
        :param data_set: The DataSet to process.
        """
        pass
