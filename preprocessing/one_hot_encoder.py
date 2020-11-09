import logging

import numpy as np

from data_structures.data_set import DataSet
from preprocessing.data_processor import DataProcessor


class OneHotEncoder(DataProcessor):
    """
    Class to encode categorical variables with one hot encoding.
    """

    def process(self, data_set: DataSet):
        """
        Normalizes the DataSet to a range of [0, 1].

        :param data_set: DataSet to process.
        """
        logging.info(f"\nOne hot encoding DataSet...")
        if len(data_set.get_categorical_attribute_indexes()) == 0:
            logging.info(f"No need to one hot encode. No categorical variables left.")
            return

        for index in data_set.get_categorical_attribute_indexes():
            column = data_set.get_attribute_column(index)

            unique = np.unique(column)
            logging.info(f"Attribute index '{index}' has categorical variables: {unique}")
            unique_dict = {x: i for i, x in enumerate(unique)}
            new_columns = [[0. for i in range(len(data_set))] for j in unique]

            for i, value in enumerate(column):
                new_columns[unique_dict[value]][i] = 1.

            for col in new_columns:
                data_set.add_column(col)

        for index in sorted(data_set.get_categorical_attribute_indexes(), reverse=True):
            data_set.remove_column(index)

        logging.info('One hot encoded DataSet:')
        logging.info(data_set.summary())

    def __repr__(self):
        return f'OneHotEncoder'
