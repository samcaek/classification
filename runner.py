import logging
import os
import sys
from typing import List
import argparse

from experiment.classification_investigator import ClassificationInvestigator
from experiment.investigator import Investigator
from knn import KNN
from decision_tree import DecisionTree
from producer import Producer
from data import preprocessed_data_sets
from data_structures.data_set import DataSet
from experiment import combine_results_files
from experiment.classification_lf import ClassificationLF
from experiment.cross_validator import CrossValidator
from experiment.loss_functioner import LossFunctioner
from utils import logging_util
from data import preprocessed_data_sets, data_set_loss_functioners, data_set_investigators



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
        """
        Runs the runner by running cross validation and writing the results to a file.
        """
        logging.info(f"\nRunning algorithm {self.name}")
        if "dtree" in self.name:
            results = CrossValidator(self.data_set, self.producer,
                                    self.investigator, self.loss_functioner, self.k_values).cross_validate_dtree()
        else: 
            results = CrossValidator(self.data_set, self.producer,
                                    self.investigator, self.loss_functioner, self.k_values).cross_validate()
        print(results)
        logging.info(f"\nAverages of loss function results:\n{results}")
        path = f"data/results/{self.name}.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as file:
            file.write(f"{repr(self)} results\n")
            if "dtree" in self.name:
                file.write("D-tree: \n\n")
                columns = ','.join([key for key in results["d-tree"]])
            else:
                file.write(f"Values of k: {', '.join([str(k) for k in self.k_values])}\n\n")
                columns = ','.join([key for key in results[self.k_values[0]]])
            file.write(columns)
            file.write('\n')
            for result_dict in results.values():
                values = [str(i) for i in result_dict.values()]
                file.write(','.join(values))
                file.write('\n')

    def __repr__(self):
        return str([self.name, self.investigator, self.k_values, self.producer, self.loss_functioner])


def main():
    parser = argparse.ArgumentParser(description='runner')

    parser.add_argument('--model-type', action="store", default="all")
    parser.add_argument('--dataset', action="store", default="iris")
    parser.add_argument('--k-values', action="store", default=[1,3,5,10,15,20,30,40,50])
    parser.add_argument('--point-threshold', action="store", default=5)
    parser.add_argument('--pi', action="store", default=0.95)

    python3 runner.py --model-type knn --dataset iris --k-values

    args = parser.parse_args()
    logging_util.start_logging()

    data_set_names = [
        'iris',
    ]

    data_set_lengths = {name: len(preprocessed_data_sets[name]) for name in data_set_names}

    runner_map = {}

    # Amounts determined from the average amount of data points produce with ENN for these data sets

    k_values = [args.k_values]

    # # Add knn runners
    if args.model_type == "all" or args.model_type == "knn":
        for name in data_set_names:
            logging.info("\n---\n")
            logging.info(f"Creating runner for KNN with {name} data set")
            runner_name = f"knn_{name}"
            runner_map[runner_name] = Runner(runner_name, k_values, preprocessed_data_sets[name], Producer(),
                                             data_set_loss_functioners[name], data_set_investigators[name])

    #Add decision tree runners
    if args.model_type == "all" or args.model_type == "decision-tree":
        for name in data_set_names:
            logging.info("\n---\n")
            logging.info(f"Creating runner for decision tree with {name} data set")
            runner_name = f"dtree_{name}"
            runner_map[runner_name] = Runner(runner_name, k_values, preprocessed_data_sets[name], DecisionTree(preprocessed_data_sets[name],
                                             size_threshold=int(args.point_threshold), pi=int(args.pi)),
                                             data_set_loss_functioners[name], data_set_investigators[name])

    logging.info('\n'.join([f'{x} : {y}' for x, y in runner_map.items()]))

    for runner in runner_map.values():
        runner.run()

    combine_results_files.main()


if __name__ == '__main__':
    main()