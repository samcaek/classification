from knn import KNN
from data_structures.data_set import DataSet


class Producer:
    """
    Class that produces a new data set from the passed one.
    """

    def __init__(self, knn: KNN = None):
        """
        Initializes Producer class.
        :param knn: The knn to use when producing a new data set.
        """
        self.knn = knn

    def produce(self, data_set: DataSet) -> DataSet:
        """
        Gets the data set. This method is meant to be overridden by child classes.
        :param data_set: The data set to produce from.
        :return: The same data set.
        """
        return data_set

    def __repr__(self):
        """
        String representation of the Producer class.
        :return: String name of class.
        """
        return 'Producer'
