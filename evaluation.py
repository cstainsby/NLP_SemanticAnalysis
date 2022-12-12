
import math 

import utils

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    confusion_matrix = [([0] * len(labels)) for i in range(len(labels))]
    for i, predicted_value in enumerate(y_pred):
        if predicted_value == 'None':
            predicted_value = '0.0'

        predicted_index = labels.index(predicted_value)
        actual_index = labels.index(y_true[i])
        confusion_matrix[actual_index][predicted_index] += 1

    return confusion_matrix 