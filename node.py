import numpy as np


class Node:
    """
    Class that implements the nodes for decision tree
    """
    def __init__(self, num_samples, num_samples_per_class, predicted_class, split_column=None, v=None,left=None, right=None, parent=None,  data_set=None):
        self.left = left
        self.right = right
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.split_column = split_column
        self.v = v
        self.parent = parent
        if not self.left and not self.right:
            self.leaf = True
        else:
            self.leaf = False
