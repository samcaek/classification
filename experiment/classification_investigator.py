import numpy as np

from experiment.investigator import Investigator


class ClassificationInvestigator(Investigator):
    """
    Class that investigates for classification.
    """

    def investigate(self, nearest_neighbors):
        """
        Gets output_value that occurs the most in the nearest neighbors.
        :param nearest_neighbors: The nearest neighbors.
        :return: Output value that occurs the most.
        """
        return np.bincount(nearest_neighbors['output_values']).argmax()

    def __repr__(self):
        return 'ClassificationInvestigator'
