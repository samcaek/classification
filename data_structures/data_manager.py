import inspect
import os
from typing import Dict

from data_structures.data_reader import DataReader
from data_structures.data_set import DataSet
from data_structures.data_writer import DataWriter


class DataManager:
    """
    Class that lets you access the data sets in a directory without having to know the relative path of the data.
    """

    def __init__(self, data_readers: Dict[str, DataReader] = None, data_paths: Dict[str, str] = None,
                 data_folder: str = ''):
        """
        Initializes a DataManager.

        :param data_readers: Dictionary mapping the name of a data set to a DataReader.
        :param data_paths: Dictionary mapping the name of a data set to the file path of the data set.
        """
        self.data_folder = data_folder
        if data_paths is None:
            data_paths = {}
        self.data_paths = {x: f'{self.data_folder}/{y}' for x, y in data_paths.items()}
        if data_readers is None:
            data_readers = {}
        self.data_readers = data_readers
        frame = inspect.stack()[1]
        self.this_dir = os.path.dirname(os.path.abspath(frame[0].f_code.co_filename))
        del frame

    def get_path(self, data_set_name):
        """
        Gets the absolute path of the data file.
        :param data_set_name: The name of the data set.
        :return: Path of data file.
        """
        folder = f'{self.data_folder}/' if self.data_folder else ''
        path = self.data_paths.get(data_set_name, f'{folder}{data_set_name}.data')
        return os.path.join(self.this_dir, path)

    def __getitem__(self, data_set_name):
        """
        Reads in the DataSet with the given name.
        :param data_set_name: Name of the DataSet.
        :return: The DataSet.
        """
        if isinstance(data_set_name, tuple):
            data_set_name = data_set_name[0] + '_' + data_set_name[1]
        data_reader = self.data_readers.get(data_set_name, DataReader())
        return data_reader.read_file(self.get_path(data_set_name))

    def __setitem__(self, data_set_name, data_set: DataSet):
        """
        Writes a DataSet using the given name.
        :param data_set_name: Name of the DataSet.
        :param data_set: DataSet to write.
        """
        if isinstance(data_set_name, tuple):
            data_set_name = data_set_name[0] + '_' + data_set_name[1]
        path = self.get_path(data_set_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        DataWriter(data_set).write_file(path)
