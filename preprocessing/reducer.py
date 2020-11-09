import logging
import math

from data_structures.data_set import DataSet
from preprocessing.data_processor import DataProcessor


class Reducer(DataProcessor):
    """
    Class to reduce the amount of data in a data set.
    """

    def __init__(self, fraction=1, max_count=100000):
        """
        Initializes a Reducer.
        :param fraction: Fraction of data to reduce to.
        """
        self.max_count = max_count - 1
        self.fraction = fraction

    def process(self, data_set: DataSet):
        """
        Reduces data set to the Reducer's fraction value.
        """
        logging.info(f"\nReducing DataSet...")
        percentage = math.ceil(len(data_set) * self.fraction)
        new_data = DataSet()
        for i in range(percentage):
            if i > self.max_count:
                break

            new_data.add(data_set.records.pop())

        data_set.records = new_data.records

        logging.info('Reduced DataSet:')
        logging.info(data_set.summary())

    def __repr__(self):
        return f'Reducer({self.fraction}, {self.max_count})'
