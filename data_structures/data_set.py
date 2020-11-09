import random
from typing import List, Tuple

import numpy as np

from utils.number_checker import NumberChecker
from data_structures.data_record import DataRecord


class DataSet:
    """
    Class that holds data.
    This data can be read in from the .data files or passed to a DataSet initializer.

    It has a list of DataRecords.
    """

    def __init__(
            self,
            records: List[DataRecord] = None,
            record_tuples: List[Tuple] = None,
            record_ndarray: np.ndarray = None,
            output_value_ndarray: np.ndarray = None
    ):
        """
        Initializes the DataSet.
        :param records: A list of DataRecords.
        """
        self.record_tuples = record_tuples
        if records is None:
            records = []

        self.records = records
        self.num_attributes = 0

        if record_tuples is not None:
            for item in record_tuples:
                self.add(DataRecord(item[0], item[1]))
        elif record_ndarray is not None and output_value_ndarray is not None:
            for record, output_value in zip(record_ndarray, output_value_ndarray):
                self.add(DataRecord(output_value, list(record)))
        elif record_ndarray is not None:
            for record in record_ndarray:
                self.add(DataRecord(None, list(record)))

    def add(self, record: DataRecord):
        """
        Adds a DataRecord to the DataSet.

        :param record: The DataRecord to add.
        """
        self.records.append(record)

    def add_column(self, column_values):
        """
        Adds another attribute to the data set.

        :param column_values: The attribute values to add.
        """
        if len(column_values) != len(self.records):
            raise ValueError(f"The length of the column values '{len(column_values)}' does not equal the length of "
                             f"the records '{len(self.records)}'.")

        for i, record in enumerate(self.records):
            value = column_values[i]
            if NumberChecker.is_float(value):
                value = float(value)

            record.attribute_values.append(value)

    def remove(self, index):
        """
        Removes the DataRecord at the given index.
        :param index: Index of DataRecord to remove.
        """
        del self.records[index]

    def remove_column(self, index):
        """
        Removes the column at the given index.

        :param index: Index of column to remove.
        """

        for record in self.records:
            del record[index]

    def get_output_values(self):
        """
        Gets a numpy array of the output values.
        :return: Numpy array of output values.
        """
        return np.array([record.output_value for record in self.records])

    def get_records_in_class(self, output_value: str):
        """
        Gets all DataRecords in the data_set that are in the specified class.
        :param output_value: A class name in the data set.
        :return: List of DataRecords.
        """
        return [record for record in self.records if record.output_value == output_value]

    def get_num_attributes(self):
        """
        Gets the number of attributes for a DataRecord in the DataSet.
        :return: Number of attributes.
        """
        return len(self.records[0])

    def get_num_rows_missing_data(self):
        """
        Gets the number of rows that have missing attribute values.
        :return Number of rows.
        """
        missing_count = 0
        for record in self.records:
            if '?' in record.attribute_values:
                missing_count += 1
        return missing_count

    def remove_records_with_missing_data(self):
        """
        Removes the DataRecords with missing data from the records list.
        """
        complete_data = []
        for record in self.records:
            if '?' not in record.attribute_values:
                complete_data.append(record)
        self.records = complete_data

    def get_attribute_column_unique(self, index: int, exclude_values=None):
        """
        Gets a list of unique values that the attributes can be in a certain column.
        :param exclude_values:
        :param index: The index of the column of attributes.
        :return: Unique valued list of attributes.
        """
        if exclude_values is None:
            exclude_values = []
        unique_attribute_values = []
        for record in self.records:
            attribute = record.attribute_values[index]
            if attribute not in exclude_values and attribute not in unique_attribute_values:
                yield attribute

    def get_attribute_column(self, index: int):
        """
        Gets a list of all the attributes in a column.
        :param index: The index of the column.
        :return: List of attributes.
        """
        return [record.attribute_values[index] for record in self.records]

    def set_column(self, index: int, attribute_values: List):
        """
        Sets a column to the given attribute values.
        :param index: Index of the column.
        :param attribute_values: New values to set the column with.
        :return: None.
        """
        if len(attribute_values) != len(self.records):
            raise ValueError(
                f"The amount of attributes '{len(attribute_values)}' must be equal to the amount of records '"
                f"{len(self.records)}'")
        attributes_index = 0
        for record in self.records:
            record.attribute_values[index] = attribute_values[attributes_index]
            attributes_index += 1

    def convert_column_to_float(self, index: int):
        """
        Converts the values in the column to floats
        :param index: The index of the column to convert
        :return: None
        """
        for record in self.records:
            record.attribute_values[index] = float(record.attribute_values[index])

    def get_num_records(self):
        """
        Returns the number of records in the DataSet.
        :return: Number of records.
        """
        return len(self.records)

    def get_continuous_attribute_indexes(self):
        """
        Gets a list of the indexes of the continuous attributes.
        :return: List of indexes.
        """
        indexes = []
        for i, value in enumerate(self.records[0].attribute_values):
            if isinstance(value, float):
                indexes.append(i)
        return indexes

    def get_categorical_attribute_indexes(self):
        """
        Gets a list of the indexes of the categorical attributes.

        :return: List of indexes.
        """
        indexes = []
        for i, value in enumerate(self.records[0].attribute_values):
            if isinstance(value, str):
                indexes.append(i)
        return indexes

    def is_regression(self):
        """
        Returns whether the data set holds regression data.
        :return: True if the data set is regression, false if not.
        """
        output_value = self.records[0].output_value
        return isinstance(output_value, float) or isinstance(output_value, int)

    def shuffle(self):
        """
        Shuffles the records randomly.
        """
        random.shuffle(self.records)

    def summary(self):
        """
        Returns a string summary of the dataset.
        :return: Summary.
        """
        if len(self.records) > 8:
            result = [',\n'.join([r.summary() for r in self.records[:3]]), '...',
                      ',\n'.join([r.summary() for r in self.records[-3:]])]
        else:
            result = [',\n'.join([r.summary() for r in self.records])]

        result.append(f"with '{self.get_num_records()}' records and '{self.get_num_attributes()}' attributes")

        return '\n'.join(result)

    def __getitem__(self, index):
        """
        Gets the DataRecord at the given index.
        :param index: Index of DataRecord.
        :return: DataRecord.
        """
        return self.records[index]

    def __iter__(self):
        """
        Returns the DataSet as an iterable.
        :return: DataSet as an iterable.
        """
        return self.records.__iter__()

    def __len__(self):
        """
        Returns how many records the DataSet has.
        :return: Length of DataSet.
        """
        return len(self.records)

    def __repr__(self):
        """
        Returns a string representation of the DataSet.
        :return: str
        """
        return ',\n'.join([repr(r) for r in self.records])

    def __eq__(self, other):
        """
        Returns whether the current object is equal to the other object.
        :param other: The object to compare with.
        :return: True if equal, False if not.
        """
        if not isinstance(other, DataSet):
            return False

        if len(self.records) != len(other.records):
            return False

        for this_record, other_record in zip(self.records, other.records):
            if this_record != other_record:
                return False

        return True
