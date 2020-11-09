import logging
import os
import sys
from typing import List

from algorithms.classification_investigator import ClassificationInvestigator
from algorithms.condensed_nn import CondensedNN
from algorithms.edited_nn import EditedNN
from algorithms.investigator import Investigator
from algorithms.k_means import KMeans
from algorithms.k_medoids import KMedoids
from algorithms.knn import KNN
from algorithms.producer import Producer
from algorithms.regression_investigator import RegressionInvestigator
from data import preprocessed_data_sets
from data_structures.data_set import DataSet
from experiment import combine_results_files
from experiment.classification_lf import ClassificationLF
from experiment.cross_validator import CrossValidator
from experiment.loss_functioner import LossFunctioner
from experiment.regression_lf import RegressionLF


class Runner:
    """
    Class that runs the experiment.
    """

    def __init__(self, name: str, k_values: List[int], data_set: DataSet, producer: Producer,
                 loss_functioner: LossFunctioner, investigator: Investigator):
        """
        Initializes an instance of Runner.
        :param k_values: Values of k to run with KNN.
        :param data_set: DataSet to run.
        :param producer: Producer used in CrossValidator.
        :param loss_functioner: LossFunctioner used in CrossValidator.
        """
        self.name = name
        self.investigator = investigator
        self.k_values = k_values
        self.data_set = data_set
        self.producer = producer
        self.loss_functioner = loss_functioner

    def run(self):
        results = []
        logging.info(
            f"\nRunning algorithm {self.name} on k values: {self.k_values}")
        for k_value in self.k_values:
            results.append(CrossValidator(KNN(self.investigator, k_value), self.data_set, self.producer,
                                          self.loss_functioner).cross_validate())
        logging.info(f"\nAverages of loss function results:\n{results}")
        path = f"../data/results/{self.name}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            file.write(f"{self.name} results\n")
            file.write(f"Values of k: {', '.join([str(k) for k in self.k_values])}\n\n")
            columns = ','.join([key for key in results[0]])
            file.write(columns)
            file.write('\n')
            for result_dict in results:
                values = [str(i) for i in result_dict.values()]
                file.write(','.join(values))
                file.write('\n')

    def __repr__(self):
        return str([self.name, self.investigator, self.k_values, self.producer, self.loss_functioner])


def main():
    # open('../logs.txt', 'w').close() # filename='../logs.txt'
    logging.basicConfig(level='INFO', format='%(message)s', stream=sys.stdout)

    data_set_names = [
         'abalone',
         'car',
        'segmentation',
         'machine',
         'forest_fires',
         'combined_wine'
    ]

    data_set_lengths = {name: len(preprocessed_data_sets[name]) for name in data_set_names}

    loss_functioners = {
        'abalone': ClassificationLF(),
        'car': ClassificationLF(),
        'segmentation': ClassificationLF(),
        'machine': RegressionLF(),
        'forest_fires': RegressionLF(),
        'combined_wine': RegressionLF()
    }

    investigators = {
        'abalone': ClassificationInvestigator(),
        'car': ClassificationInvestigator(),
        'segmentation': ClassificationInvestigator(),
        'machine': RegressionInvestigator(),
        'forest_fires': RegressionInvestigator(),
        'combined_wine': RegressionInvestigator()
    }

    runner_map = {}

    # Amounts determined from the average amount of data points produce with ENN for these data sets
    classification_data_set_cluster_amounts = {
        'abalone': 1,
        'car': 200,
        'segmentation': 6
    }

    k_values = [1, 9, 21]
    #k_values = [1]

    # Add knn runners
    for name in data_set_names:
        logging.info("\n---\n")
        logging.info(f"Creating runner for KNN with {name} data set")
        runner_name = f"knn_{name}"
        runner_map[runner_name] = Runner(runner_name, k_values, preprocessed_data_sets[name], Producer(),
                                         loss_functioners[
                                             name], investigators[name])

    # runner_map['knn_segmentation'].run()

    # Add enn runners
    for name in data_set_names[:3]:
        logging.info("\n---\n")
        logging.info(f"Creating runner for EditedNN with {name} data set")
        data_set = preprocessed_data_sets[name]
        edited_nn = EditedNN(KNN(ClassificationInvestigator()))
        edited_nn.extract_validation_set(data_set, 0.1)
        runner_name = f"enn_{name}"
        runner_map[runner_name] = Runner(runner_name, k_values, data_set, edited_nn,
                                         loss_functioners[name], investigators[name])

    # runner_map[f'enn_segmentation'].run()

    # Add cnn runners
    for name in data_set_names[:3]:
        logging.info("\n---\n")
        logging.info(f"Creating runner for CondensedNN with {name} data set")
        runner_name = f"cnn_{name}"
        runner_map[runner_name] = Runner(runner_name, k_values, preprocessed_data_sets[name], CondensedNN(
            KNN(ClassificationInvestigator())),
                                         loss_functioners[
                                             name], investigators[name])

    # runner_map[f'cnn_segmentation'].run()

    # Add k-means classification runners
    for name in data_set_names[:3]:
        cluster_amount = classification_data_set_cluster_amounts[name]
        logging.info("\n---\n")
        logging.info(f"Creating runner for K Means with {name} data set with {cluster_amount} clusters")
        runner_name = f"kmeans_{name}"
        runner_map[runner_name] = Runner(runner_name, k_values, preprocessed_data_sets[name], KMeans(
            KNN(investigators[name]), cluster_amount),
                                         loss_functioners[
                                             name], investigators[name])

    # runner_map['kmeans_segmentation'].run()

    # Add k-medoids classification runners
    for name in data_set_names[:3]:
        cluster_amount = classification_data_set_cluster_amounts[name]
        logging.info("\n---\n")
        logging.info(f"Creating runner for K Medoids with {name} data set with {cluster_amount} clusters")
        runner_name = f"kmedoids_{name}"
        runner_map[runner_name] = Runner(runner_name, k_values, preprocessed_data_sets[name], KMedoids(
            KNN(investigators[name]), cluster_amount),
                                         loss_functioners[
                                             name], investigators[name])

    # runner_map['kmedoids_segmentation'].run()

    for name in data_set_names[3:]:
        runner_name = f"kmeans_{name}"
        cluster_amount = data_set_lengths[name] // 4
        logging.info("\n---\n")
        logging.info(f"Creating runner for K Means with {name} data set with {cluster_amount} clusters")
        runner_map[runner_name] = Runner(runner_name, k_values, preprocessed_data_sets[name], KMeans(
            KNN(investigators[name]), cluster_amount),
                                         loss_functioners[
                                             name], investigators[name])

        logging.info("\n---\n")
        logging.info(f"Creating runner for K Medoids with {name} data set with {cluster_amount} clusters")
        runner_name = f"kmedoids_{name}"
        runner_map[runner_name] = Runner(runner_name, k_values, preprocessed_data_sets[name], KMedoids(
            KNN(investigators[name]), cluster_amount),
                                         loss_functioners[
                                             name], investigators[name])

    logging.info('\n'.join([f'{x} : {y}' for x, y in runner_map.items()]))

    runner_map['knn_combined_wine'].run()

    combine_results_files.main()



if __name__ == '__main__':
    main()
