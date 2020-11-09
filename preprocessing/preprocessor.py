import logging
import random
import sys
from typing import List, Dict

from data import original_data_sets, preprocessed_data_sets
from data_structures.data_set import DataSet
from preprocessing.class_joiner import ClassJoiner
from preprocessing.class_scaler import ClassScaler
from preprocessing.data_processor import DataProcessor
from preprocessing.normalizer import Normalizer
from preprocessing.one_hot_encoder import OneHotEncoder
from preprocessing.circular_encoder import CircularEncoder
from preprocessing.reducer import Reducer
from preprocessing.scaler import Scaler
from preprocessing.shuffler import Shuffler
from utils import logging_util


class Preprocessor:
    """
    Class that pre-processes a DataSet using a list of DataProcessors.
    """

    def __init__(self, data_processors: List[DataProcessor]):
        """
        Initializes a Preprocessor.
        :param data_processors: List of DataProcessors to be used.
        """
        self.data_processors = data_processors

    def preprocess(self, data_set: DataSet):
        """
        Pre-processes the DataSet.
        :param data_set: The DataSet to pre-process.
        """
        logging.info(f'Preprocessing')
        for data_processor in self.data_processors:
            data_processor.process(data_set)

    def __repr__(self):
        processors = ', '.join(repr(x) for x in self.data_processors)
        return f'Preprocessor({processors})'


processor_map: Dict[str, List[DataProcessor]] = {
    'iris': [
        ClassScaler()
    ],
    'car': [
        ClassScaler(),
        Scaler(index_value_map={
            0: ['vhigh', 'high', 'med', 'low'],
            1: ['vhigh', 'high', 'med', 'low'],
            2: [2.0, 3.0, 4.0, '5more'],
            3: [2.0, 4.0, 'more'],
            4: ['small', 'med', 'big'],
            5: ['low', 'med', 'high']
        })
    ]
}

additional_processors = [Normalizer(), Shuffler()]

processor_map = {x: y + additional_processors for x, y in processor_map.items()}

sort_order = {
    Scaler: 1,
    CircularEncoder: 2,
    OneHotEncoder: 3,
    Normalizer: 4,
    Shuffler: 5,
    ClassJoiner: 8,
    ClassScaler: 9,
}

processor_map = {x: sorted(y, key=lambda p: sort_order[type(p)]) for x, y in processor_map.items()}

preprocessor_map = {x: Preprocessor(y) for x, y in processor_map.items()}


def main(data_sets_to_process=None):
    random.seed(123)
    logging_util.start_logging()
    logging.info('Preprocessor dictionary:')
    logging.info('\n'.join([f"'{x}' : {y}" for x, y in processor_map.items()]))

    if data_sets_to_process is not None:
        pass
    elif len(sys.argv) > 1:
        data_sets_to_process = [sys.argv[1]]
    else:
        data_sets_to_process = processor_map

    for data_set_name in data_sets_to_process:
        data_set = original_data_sets[data_set_name]
        preprocessor_map[data_set_name].preprocess(data_set)
        preprocessed_data_sets[data_set_name] = data_set


if __name__ == '__main__':
    main()
