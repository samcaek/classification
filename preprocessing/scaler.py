import logging
from typing import List, Dict

from data_structures.data_set import DataSet
from preprocessing.data_processor import DataProcessor


class Scaler(DataProcessor):
    """
    Class to scale data from a sequential list of categorical attribute value strings to float values.
    """

    def __init__(self, index_value_map: Dict[int, List[str]]):
        """
        Initializes a Scaler.
        :param index_value_map: Map of attribute index to their list of attribute values.
        """
        self.index_value_map = {x: {index_value_map[x][i]: float(i) for i in range(len(index_value_map[x]))} for x in
                                index_value_map}

    def process(self, data_set: DataSet):
        """
        Scales data from the sequential list of categorical attribute value strings to float values.
        """
        logging.info('\nScaling the attributes in the DataSet...')
        map_as_string = '\n'.join(f'{x}: {y}' for x, y in self.index_value_map.items())
        logging.info(f'Using map:\n{map_as_string}')
        for record in data_set:
            for index in self.index_value_map:
                record[index] = self.index_value_map[index][record[index]]
        logging.info('Scaled DataSet:')
        logging.info(data_set.summary())

    def __repr__(self):
        return f'Scaler({self.index_value_map})'
