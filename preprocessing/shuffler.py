import logging

from data_structures.data_set import DataSet
from preprocessing.data_processor import DataProcessor


class Shuffler(DataProcessor):
    """
    Class to shuffle data.
    """

    def process(self, data_set: DataSet):
        """
        Shuffles the data set.
        :param data_set: DataSet to shuffle.
        """
        logging.info('\nShuffling DataSet...')
        data_set.shuffle()
        logging.info('Shuffled DataSet:')
        logging.info(data_set.summary())

    def __repr__(self):
        return 'Shuffler'
