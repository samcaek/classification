from time import time
from typing import List, Tuple, Dict

import numpy as np

from experiment.investigator import Investigator
from data_structures.data_record import DataRecord
from data_structures.data_set import DataSet


class KNN:
    """
    Class that implements the KNN algorithm.
    """

    def __init__(self, investigator: Investigator, k: int = 1):
        self.investigator = investigator
        self.k = k

    def get_nearest_neighbors(
            self,
            data_set: np.ndarray,
            record: np.ndarray,
            output_values: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Gets the k nearest neighbors.
        :param data_set: Array of records to find nearest neighbors in.
        :param record: Record to find nearest neighbor for.
        :param output_values: Output values of the records.
        :return: Dictionary of the nearest records, the output values, and the distances.
        """
        distances = np.sqrt(np.sum((data_set - record) ** 2, axis=1))
        partition_size = self.k - 1 if self.k < len(distances) else len(distances) - 1
        partition_indexes = np.argpartition(distances, partition_size)
        k_nearest_indexes = partition_indexes[:self.k]

        return {
            'records': data_set[k_nearest_indexes],
            'output_values': output_values[k_nearest_indexes],
            'distances': distances[k_nearest_indexes]
        }

    def analyze(self, data_set: np.ndarray, record: np.ndarray, output_values: np.ndarray):
        """
        Analyzes the record by finding its nearest neighbors and investigating them.
        :param data_set: Array of data to analyze with.
        :param output_values: Output values of the data.
        :param record: Record to analyze.
        :return: Predicted output value.
        """
        nearest_neighbors = self.get_nearest_neighbors(data_set, record,
                                                       output_values)
        return self.investigator.investigate(nearest_neighbors)