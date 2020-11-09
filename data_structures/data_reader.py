import logging

from data_structures.data_record import DataRecord
from data_structures.data_set import DataSet
from utils.number_checker import NumberChecker


class DataReader:
    """
    Class that reads data from a file to a DataSet.
    """

    def __init__(self, output_value_index=0, ignore_attributes=None, separator=',', is_regression=False,
                 has_column_names=False):
        """
        Initializes a DataReader.

        :param output_value_index: Attribute index of the output value.
        :param ignore_attributes: Indexes of attributes to ignore.
        :param separator: What character the data is separated with.
        :param is_regression: Whether the data is regression data.
        :param has_column_names: Whether the data has column names.
        """
        if ignore_attributes is None:
            ignore_attributes = []

        self.separator = separator
        self.has_column_names = has_column_names
        self.is_regression = is_regression
        self.remove_attributes = ignore_attributes
        self.remove_attributes.append(output_value_index)
        self.output_value_index = output_value_index

    def read_file(self, filename) -> DataSet:
        """
        Reads data from a file into a DataSet object.
        :return: The DataSet created from the file.
        """
        data_set = DataSet()
        remove_first_column = self.has_column_names
        #logging.info(f"\nReading in file '{filename}' to DataSet")

        with open(filename, "r") as file:
            while True:
                record_string = file.readline().rstrip()
                if remove_first_column:
                    remove_first_column = False
                    continue

                if record_string == '':
                    break

                parsed_record = record_string.split(self.separator)
                output_value = parsed_record[self.output_value_index]

                for index in sorted(self.remove_attributes, reverse=True):
                    del parsed_record[index]

                for i in range(len(parsed_record)):
                    if NumberChecker.is_float(parsed_record[i]):
                        # noinspection PyTypeChecker
                        parsed_record[i] = round(float(parsed_record[i]), 10)

                if self.is_regression:
                    output_value = float(output_value)
                elif NumberChecker.is_int(output_value):
                    output_value = int(output_value)

                data_set.add(DataRecord(output_value, parsed_record))

        #logging.info('DataSet read:')
        #logging.info(data_set.summary())
        return data_set
