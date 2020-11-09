import logging
from typing import List, Dict

import numpy as np

from data_structures.data_set import DataSet
from preprocessing.data_processor import DataProcessor


class ClassJoiner(DataProcessor):
    """
    Class to join class output values together into groups which then define the classes.
    """

    def __init__(self, join_on):
        """
        Initializes a ClassJoiner.

        :param join_on: A 2D list of what class values should be joined.
        """
        self.join_on = join_on

    def process(self, data_set: DataSet):
        """
        Joins class values together.

        :param data_set: DataSet to process.
        :return: None
        """
        logging.info('\nJoining the classes in the DataSet...')
        if not np.array_equal(np.concatenate(self.join_on), np.unique(data_set.get_output_values())):
            raise ValueError(f'The join_on lists {self.join_on} do not match the output values '
                             f'{np.unique(data_set.get_output_values())}')

        class_dict = {}
        for i, classes in enumerate(self.join_on):
            for class_name in classes:
                class_dict[class_name] = i

        logging.info(f'Class names are mapped with {self.join_on}')
        for record in data_set:
            record.output_value = class_dict[record.output_value]

        logging.info('Class joined DataSet:')
        logging.info(data_set.summary())

    def __repr__(self):
        return f'ClassJoiner({self.join_on})'
