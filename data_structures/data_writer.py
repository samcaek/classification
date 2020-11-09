import logging

from data_structures.data_set import DataSet


class DataWriter:
    """
    Class that writes a DataSet to a file.
    """

    def __init__(self, data_set: DataSet, output_value_index=0, ignore_attributes=None):
        """
        Initializes a DataWriter.
        :param data_set: The DataSet to write.
        :param output_value_index: The index of the class.
        :param ignore_attributes: Which attribute indexes to ignore.
        """
        if ignore_attributes is None:
            ignore_attributes = []
        self.data_set = data_set
        self.output_value_index = output_value_index

        self.insert_indexes = [(i, 'ignore') for i in ignore_attributes]
        self.insert_indexes.append((output_value_index, 'classname'))
        self.insert_indexes = sorted(self.insert_indexes)

    def write_file(self, filename):
        """
        Writes the contained DataSet to a file.
        :param filename: Name of file to write to.
        """
        logging.info(f"\nWriting DataSet to file '{filename}'")
        with open(filename, 'w') as file:
            for record in self.data_set.records:
                line = record.attribute_values.copy()
                for index, value in self.insert_indexes:
                    if value == 'classname':
                        line.insert(index, record.output_value)
                    else:
                        line.insert(index, value)
                file.write(','.join([str(x) for x in line]) + '\n')
