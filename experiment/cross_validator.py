import logging
from typing import List, Tuple, Dict

import numpy as np

from knn import KNN
from decision_tree import DecisionTree
from producer import Producer
from data_structures.data_record import DataRecord
from data_structures.data_set import DataSet
from experiment.loss_functioner import LossFunctioner



class CrossValidator:
    """
    Class used to run n-fold cross validation on a DataSet.
    """

    def __init__(self, data_set: DataSet, producer: Producer, investigator, loss_functioner: LossFunctioner,
                 k_values: List[int], folds: int = 10):
        """
        Initializes an instance of CrossValidator.
        :param data_set: The data set to run cross validation with.
        :param producer: The producer that will use the algorithm to produce our augmented data set.
        :param investigator: Investigator to analyze results with.
        :param loss_functioner: Loss functions to run on results.
        :param k_values: List of values to use for k in our KNN.
        :param folds: Number of folds to use for cross validation.
        """
        self.data_set = data_set
        self.producer = producer
        self.loss_functioner = loss_functioner
        self.folds = folds
        self.k_values = k_values
        self.knns = [KNN(investigator, k) for k in k_values]
        self.dtree = DecisionTree(data_set)

    def cross_validate(self) -> Dict[str, float]:
        """
        Runs cross validation and then runs loss functions on the result.
        :return: Dictionary of loss function names and their result.
        """
        logging.info(f"\nCross Validating with {self.folds} folds:")
        results = self.get_results()
        for k, result in results.items():
            loss_function_results = self.loss_functioner.run_loss_functions(result)
            results[k] = loss_function_results
        return results

    def cross_validate_dtree(self) -> Dict[str, float]:
        """
        Runs cross validation and then runs loss functions on the result.
        :return: Dictionary of loss function names and their result.
        """
        logging.info(f"\nCross Validating with {self.folds} folds:")
        results = self.get_results_dtree()
        for k, result in results.items():
            loss_function_results = self.loss_functioner.run_loss_functions(result)
            results[k] = loss_function_results
        return results

    def get_results(self):
        """
        Gets results from running cross validation.
        :return: A list of length of num_bins containing tuples of predicted output values and actual output values.
        """
        record_bins = self.separate_records()
        data_sets = self.get_data_sets_with_testing_records(record_bins)
        results = {k: [] for k in self.k_values}
        for i, validation_data in enumerate(data_sets):
            logging.info(f"\nFold {i + 1}/{self.folds}\n")
            produced_set = self.producer.produce(validation_data[0])
            produced_output_values = produced_set.get_output_values()
            produced_set = np.array(produced_set)

            logging.info(f"\nRunning KNN on {len(validation_data[1])} testing records to get predicted values:")

            for knn in self.knns:
                predicted_output_values = []
                actual_output_values = []

                for record in validation_data[1]:
                    predicted_output_values.append(knn.analyze(produced_set, np.array(record), produced_output_values))
                    actual_output_values.append(record.output_value)
                results[knn.k].append((predicted_output_values, actual_output_values))

            logging.info(f"Predicted values:\n{predicted_output_values}\nActual values:\n{actual_output_values}")

        logging.info(f"\n\nFinal {self.folds} fold prediction results:")
        logging.info(results)

        return results

    def get_results_dtree(self):
        """
        Gets results from running cross validation.
        :return: A list of length of num_bins containing tuples of predicted output values and actual output values.
        """
        record_bins = self.separate_records()
        data_sets = self.get_data_sets_with_testing_records(record_bins)
        results = {"d-tree": []}
        for i, validation_data in enumerate(data_sets):
            logging.info(f"\nFold {i + 1}/{self.folds}\n")
            produced_set = validation_data[0]
            produced_output_values = produced_set.get_output_values()
            produced_set = np.array(produced_set)

            logging.info(f"\nRunning Decision Tree on {len(validation_data[1])} testing records to get predicted values:")

            predicted_output_values = []
            actual_output_values = []
            for record in validation_data[1]:
                predicted_output_values.append(self.dtree.predict(np.array(record)))
                actual_output_values.append(record.output_value)
                
            results["d-tree"].append((predicted_output_values, actual_output_values))

            logging.info(f"Predicted values:\n{predicted_output_values}\nActual values:\n{actual_output_values}")

        logging.info(f"\n\nFinal {self.folds} fold prediction results:")
        logging.info(results)

        return results

    def separate_records(self) -> List[List[DataRecord]]:
        """
        Separates records into the number of bins to cross validate.
        :return: A matrix of DataRecords in k bins.
        """
        num_records = len(self.data_set.records)
        # Calculate bin size so we know how many records each bin will hold
        bin_size = num_records / self.folds
        record_matrix = [[] for _ in range(self.folds)]
        for i, record in enumerate(self.data_set.records):
            bin_index = int(i / bin_size)
            if bin_index == self.folds:
                bin_index = self.folds - 1
            record_matrix[bin_index].append(record)
        return record_matrix

    def get_data_sets_with_testing_records(self, record_bins: List[List[DataRecord]]):
        """
        Gets the DataSets and the DataRecords to classify.
        :param record_bins: Matrix of DataRecords in k bins.
        :return: List of DataSets and DataRecords.
        """
        data_sets = []
        for i in range(self.folds):
            record_bins_copy = record_bins.copy()
            record_bins_copy.pop(i)
            training_records = [record for record_bin in record_bins_copy for record in record_bin]
            data_sets.append((DataSet(records=training_records), record_bins[i]))
        return data_sets