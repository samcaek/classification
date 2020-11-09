from typing import List, Tuple, Dict

from experiment.loss_functioner import LossFunctioner


class ClassificationLF(LossFunctioner):
    """
    Classification Loss Functioner. Class to calculate loss functions for classification data sets.
    """

    def run_loss_functions(self, results: List[Tuple[List, List]]) -> Dict[str, float]:
        """
        Calculates the average loss function values for the list of predicted classes vs observed classes.

        :param results: List of predicted classes and observed classes.
        :return: Dictionary mapping the name of the loss function to the average loss function value.
        """
        return self.compute_loss_results(results, self.multi_class_loss_functions)

    @staticmethod
    def multi_class_loss_functions(y_pred, y_obs):
        """
        Gets a dictionary of loss function values.
        Including:
        precision micro and macro,
        recall micro and macro,
        accuracy average,
        and f1.
        :param y_pred: The list of predicted classes.
        :param y_obs: The list of the observed (actual) classes.
        :return: Dictionary of loss function results.
        """
        classes = set(y_pred + y_obs)
        sum_tp, sum_tn, sum_fp, sum_fn, acc_sum = 0, 0, 0, 0, 0

        for c in classes:
            tp_i = sum([1 if y_pred[x] == c and y_obs[x] == c else 0 for x in range(len(y_pred))])
            tn_i = sum([1 if y_pred[x] != c and y_obs[x] != c else 0 for x in range(len(y_pred))])
            fp_i = sum([1 if y_pred[x] == c and y_obs[x] != c else 0 for x in range(len(y_pred))])
            fn_i = sum([1 if y_pred[x] != c and y_obs[x] == c else 0 for x in range(len(y_pred))])

            sum_tp += tp_i
            sum_fp += fp_i
            sum_fn += fn_i
            sum_tn += tn_i

            acc_sum += ((tp_i + tn_i) / (tp_i + tn_i + fn_i + fp_i))

        # end for

    
        acc = acc_sum / len(classes)

        loss_function_dict = {"avg_accuracy": acc}

        return loss_function_dict

    def __repr__(self):
        return 'ClassificationLF'
