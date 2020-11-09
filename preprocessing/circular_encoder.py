import logging

import numpy as np

from data_structures.data_set import DataSet
from preprocessing.data_processor import DataProcessor
from typing import List, Dict


class CircularEncoder(DataProcessor):
    """
    Class used to circularly encode categorical variables. These are usually variables based on time, such as days or
    months.
    """

    def __init__(self, attribute_indexes=List[int]):
        """
        Initializes a CircularEncoder.

        :param attribute_indexes: The attribute indexes which should be circular encoded.
        """
        self.attribute_indexes = attribute_indexes

    def process(self, data_set: DataSet):
        """
        The process will take numerical data and change it to sin_cos data.

        :param data_set: DataSet to process.
        """

        logging.info("\nCircular Encoding Data...")

        for index in self.attribute_indexes:
            column = data_set.get_attribute_column(index)
            max_value = max(column)
            mult = (2 * np.pi) / (max_value + 1)
            sin_column = []
            cos_column = []
            for value in column:
                sin_column.append(np.sin(mult * value))
                cos_column.append(np.cos(mult * value))

            data_set.add_column(sin_column)
            data_set.add_column(cos_column)

        for index in sorted(self.attribute_indexes, reverse=True):
            data_set.remove_column(index)

        logging.info('Circular encoded data:')
        logging.info(data_set.summary())

    def __repr__(self):
        return f'CircularEncoder({self.attribute_indexes})'
