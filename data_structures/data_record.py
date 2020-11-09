from typing import List


class DataRecord:
    """
    Class that stores a class name, and a list of attribute values.
    """

    def __init__(self, output_value, attribute_values: List):
        """
        Initializes a DataRecord.
        :param output_value: The record's output value. Either a class or a regression value.
        :param attribute_values: List of attribute values.
        """
        self.output_value = output_value
        self.attribute_values = attribute_values

    def summary(self):
        """
        Returns a summary of the DataRecord with rounded float values.
        :return: Summary of DataRecord.
        """
        attribute_values = [round(x, 3) if isinstance(x, float) else x for x in self.attribute_values]
        return f'({repr(self.output_value)}, {attribute_values})'

    def __getitem__(self, index):
        """
        Gets the attribute value at the given index.
        :param index: Index of value.
        :return: Attribute value.
        """
        return self.attribute_values[index]

    def __setitem__(self, index, value):
        """
        Sets the attribute value at the given index.
        :param index: Index of value.
        :param value: Value to set.
        """
        self.attribute_values[index] = value

    def __delitem__(self, index):
        """
        Removes the attribute value at the given index.

        :param index: Attribute value index to remove.
        """

        del self.attribute_values[index]

    def __len__(self):
        return len(self.attribute_values)

    def __str__(self):
        """
        A string representation of a DataRecord.
        :return: str
        """
        return f'({repr(self.output_value)}, {self.attribute_values})'

    def __repr__(self):
        """
        A string representation of a DataRecord.
        :return: str
        """
        return self.__str__()

    def __eq__(self, other):
        """
        Returns whether this object is equal to the other object.
        :param other: The other object.
        :return: True if equal, false if not.
        """
        if not isinstance(other, DataRecord):
            return False

        if self.attribute_values != other.attribute_values:
            return False

        if self.output_value != other.output_value:
            return False

        return True

    def __hash__(self):
        """
        Returns the hashed value of the string representation.
        :return: Hashed representation.
        """
        return hash(self.__repr__())
