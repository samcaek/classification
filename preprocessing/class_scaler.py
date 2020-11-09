import logging
from typing import List, Dict

import numpy as np

from data_structures.data_set import DataSet
from preprocessing.data_processor import DataProcessor


class ClassScaler(DataProcessor):
    """
    Class to assign class output values to integers.
    """

    def process(self, data_set: DataSet):
        """
        Assigns class output values to integers.

        :param data_set: DataSet to process.
        :return: None
        """
        logging.info('\nScaling the classes in the DataSet...')
        classes = data_set.get_output_values()
        unique_classes = sorted(list(set(classes)))
        class_dict = {class_name: i for i, class_name in enumerate(unique_classes)}
        logging.info(f'Amount of different classes: {len(class_dict)}')
        logging.info(f'Class names are mapped with {class_dict}')
        for record in data_set:
            record.output_value = class_dict[record.output_value]

        logging.info('Class Scaled DataSet:')
        logging.info(data_set.summary())

    def __repr__(self):
        return 'ClassScaler'
