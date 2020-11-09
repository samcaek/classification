class NumberChecker:
    """
    Utility class to check numbers.
    """

    @staticmethod
    def is_float(value):
        """
        Checks if the value is a float.

        :param value: The value to check.
        :return: True if the value is a float, False if not.
        """
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def is_int(value):
        """
        Checks if the value is an int.

        :param value: The value to check
        :return: True if the value is an int, False otherwise.
        """
        try:
            int(value)
            return True
        except ValueError:
            return False
