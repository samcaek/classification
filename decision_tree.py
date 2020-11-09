from time import time
from typing import List, Tuple, Dict

import numpy as np

from data_structures.data_record import DataRecord
from data_structures.data_set import DataSet
from node import Node
import math

class DecisionTree:
    """
    Class that implements the KNN algorithm.
    """

    def __init__(self, data_set: DataSet, size_threshold=5, pi=0.4):
        self.size_threshold = size_threshold
        self.pi = pi
        self.data_set = data_set

        output_vals = data_set.get_output_values()
        self.classes = list(set(output_vals))
        n_i = {i: (output_vals == i).sum() for i in self.classes}
        self.parent_node = None

        D = np.array(data_set)
        D = np.insert(D, 4, values=output_vals, axis=1)
        self.dtree = self.decision_tree(D, size_threshold, pi, self.parent_node, None)



    def decision_tree(self, data_set, size_threshold, pi, parent_node, direction):
        # Function implements decision tree algo (19.1) from https://dataminingbook.info/book_html/chap19/book-watermark.html
        n = len(data_set)
        output_vals = data_set[:,4]
        D = data_set
        num_attributes = len(D[1])
        classes = list(set(output_vals))
        n_i = {i: (output_vals == i).sum() for i in self.classes}
    
        purity = self.purity(n_i)

        if n <= size_threshold or purity >= pi:
            c = max(n_i, key=n_i.get)
            leaf_node = Node(n, n_i, c, parent=parent_node)

            if direction == "left":
                parent_node.left = leaf_node
            elif direction == "right":
                parent_node.right = leaf_node

            return 
        
        split_point = None
        max_score = 0
        v_final = 0
        split_col = 0
        for i in range(num_attributes-1):
            column = D[:,i]
            v, score = self.evaluate_numeric_attribute(D, column, i)
            #print("v is %s -- score is %s" % (v, score))
            if score > max_score:
                split_point = len(np.argwhere(column <= v))
                max_score = score
                split_col = i
                v_final = v


        sorted_D = D[D[:,split_col].argsort()]
        # print(sorted_D)
        # print(split_col)
        D_Y = sorted_D[split_point:len(D)]
        D_N = sorted_D[0:split_point]

        D_Y_out = D_Y[:,4]
        classes_Y = list(set(D_Y_out))
        D_N_out = D_N[:,4]
        classes_N = list(set(D_N_out))

        n_i_Y = {i: (D_Y_out == i).sum() for i in classes_Y}
        n_i_N = {i: (D_N_out == i).sum() for i in classes_N}

        left_node = Node(len(D_Y), n_i_Y, None, data_set=D_Y)
        right_node = Node(len(D_N), n_i_N, None, data_set=D_N)

        node = Node(n, n_i, None, split_column=split_col, v=v_final , left=left_node, right=right_node, parent=parent_node, data_set=D)
        if not self.parent_node:
            self.parent_node = node

        left_node.parent = node
        right_node.parent = node


        self.decision_tree(D_Y, self.size_threshold, self.pi, node, "left")
        self.decision_tree(D_N, self.size_threshold, self.pi, node, "right")



    def evaluate_numeric_attribute(self, D, X_j, attribute_col):
        D = D[D[:,attribute_col].argsort()]
        n = len(D)
        #print(D)
        midpoints = []
        num_attributes = len(D[1]-1)
        classes = list(set(D[:,4]))
        n_i = {i:0 for i in classes}
        N_vi = {}
        for j in range(len(D)-1):
            n_i[D[j,4]] = n_i.get(D[j,4]) + 1

            if j < len(D):
                if X_j[j+1] != X_j[j]:
                    v = ( X_j[j+1] + X_j[j] ) / 2
                    N_vi[v] = N_vi.get(v,{})
                    midpoints = midpoints + [v]
                    for i in range(len(classes)):
                        N_vi[v][i] = n_i.get(i,0)

        n_i[D[n-1, 4]] = n_i.get(D[n-1, 4]) + 1

        v_star = 0
        max_score = 0
        for v in list(set(midpoints)):
            split_point = min(np.argwhere(D[:,attribute_col] > v))[0]
            for i in range(len(classes)):
                p_ci_DY = N_vi[v][i] / sum([N_vi[v][j] for j in range(len(classes))])
                p_cu_DN = ( n_i[i] - N_vi[v][i] ) / sum([n_i[j] - N_vi[v][j] for j in range(len(classes))])
            D_Y = D[split_point:len(D)]
            D_N = D[0:split_point]
            current_score = gain(D, D_Y, D_N)
            if current_score > max_score:
                max_score = current_score
                v_star = v
            


        return v_star, max_score


    def predict(self, record: np.ndarray):
        """
        Analyzes the record by finding its nearest neighbors and investigating them.
        :param data_set: Array of data to analyze with.
        :param output_values: Output values of the data.
        :param record: Record to analyze.
        :return: Predicted output value.
        """
        current_node = self.parent_node
        while not current_node.leaf:
            if record[current_node.split_column] > current_node.v:
                current_node = current_node.left
            else:
                current_node = current_node.right

        return current_node.predicted_class


    def purity(self, n_i):
        maximum = max(n_i, key=n_i.get)
        return n_i[maximum] / sum([n_i.get(i) for i in n_i.keys()])

def entropy(D):
    output_vals = D[:,4]
    classes = list(set(output_vals))

    n_i = {i: (output_vals == i).sum() for i in classes}

    sum_list = []
    for j in n_i.keys():
        p_ci_given_D = (n_i[j] / sum([n_i.get(i) for i in n_i.keys()]))
        current_entropy = p_ci_given_D * math.log(p_ci_given_D,2 )
        sum_list.append(current_entropy)

    return -1*sum(sum_list)

def split_entropy(D_Y,D_N):
    return ( 
        ((len(D_Y) / (len(D_Y) + len(D_N))) * entropy(D_Y)) + ((len(D_N) / (len(D_Y) + len(D_N))) * entropy(D_N))
    )

def gain(D, D_Y, D_N):
    return entropy(D) - split_entropy(D_Y, D_N)


