import logging

from data_structures.data_set import DataSet
from preprocessing.data_processor import DataProcessor


class Normalizer(DataProcessor):
    """
    Class to normalize continuous attribute values to range of [0, 1].
    """

    @staticmethod
    def normalize_value(value, min_val, max_val):
        """
        Normalizes a value.
        :param value: The value to normalize.
        :param min_val: The minimum value that this attribute was in the DataSet.
        :param max_val: The maximum value that this attribute was in the DataSet.
        :return: None.
        """
        if max_val - min_val == 0:
            return 0.0
        return (value - min_val) / (max_val - min_val)

    def process(self, data_set: DataSet):
        """
        Normalizes the DataSet to a range of [0, 1].

        :param data_set: DataSet to process.
        :return: None.
        """
        logging.info(f"\nNormalizing DataSet...")
        for index in data_set.get_continuous_attribute_indexes():
            column = data_set.get_attribute_column(index)
            min_val = min(column)
            max_val = max(column)
            if min_val == 0.0 and max_val == 1.0:
                continue

            normalized_column = []
            for value in column:
                normalized_column.append(self.normalize_value(value, min_val, max_val))

            data_set.set_column(index, normalized_column)
        logging.info('Normalized DataSet:')
        logging.info(data_set.summary())

    def __repr__(self):
        return f'Normalizer'
