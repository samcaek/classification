from experiment.classification_investigator import ClassificationInvestigator
from data_structures.data_manager import DataManager
from data_structures.data_reader import DataReader
from experiment.classification_lf import ClassificationLF

data_set_names = [
    'iris',
]

data_set_is_classification = {
    'iris': True,
}

data_set_loss_functioners = {x: ClassificationLF() if data_set_is_classification[x] else RegressionLF() for x in
                             data_set_names}

data_set_investigators = {x: ClassificationInvestigator() if data_set_is_classification[x] else RegressionInvestigator()
                          for x in
                          data_set_names}

original_data_readers = {
    'iris': DataReader(-1),
}

data_paths = {x: f'{x}_data/{x}.data' for x in original_data_readers.keys()}

original_data_sets = DataManager(original_data_readers, data_paths, 'original')

preprocessed_data_readers = {
    'iris': DataReader(),
}

preprocessed_data_sets = DataManager(preprocessed_data_readers, data_paths, 'preprocessed')