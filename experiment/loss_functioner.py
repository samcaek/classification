import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Callable, Dict


class LossFunctioner(ABC):
    """
    An abstract class runs loss functions on predicted values vs actual values.
    """

    @abstractmethod
    def run_loss_functions(self, results: List[Tuple[List[str], List[str]]]):
        """
        Runs loss functions on a list of predicted output values vs actual output values.
        :param results: A list of predicted output values and actual output values.
        :return: Dictionary of loss functions
        """
        pass

    @staticmethod
    def compute_loss_results(results: List[Tuple[List[str], List[str]]],
                             calculate_loss: Callable[[List, List], Dict[str, float]]):
        first_loss = calculate_loss(results[0][0], results[0][1])
        averages = {x: 0 for x in first_loss}

        loss_values_list = []

        for result_tuple in results:
            loss_values = calculate_loss(result_tuple[0], result_tuple[1])
            loss_values_list.append(loss_values)
            averages = {x: averages[x] + loss_values[x] for x in averages}

        averages = {x: averages[x] / len(results) for x in averages}

        standard_deviations = {x: 0 for x in first_loss}
        for loss_values in loss_values_list:
            for loss_method, loss_value in loss_values.items():
                diff = loss_value - averages[loss_method]
                standard_deviations[loss_method] += pow(diff, 2)

        for loss_method, deviation in standard_deviations.items():
            divide_by_n = deviation / len(loss_values_list)
            standard_deviations[loss_method] = math.sqrt(divide_by_n)

        standard_deviations = {f'{x}_sd': y for x, y in standard_deviations.items()}
        averages.update(standard_deviations)

        return averages
