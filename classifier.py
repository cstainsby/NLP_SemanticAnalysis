
import math
import numpy as np
import utils


# from mysklearn import myevaluation, myutils
# from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
# import mysklearn

class NaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(dict of key:str and value:float(P)): The prior probabilities computed for each
            label in the training set.
        posteriors(list of dict of key:attribute value and value of dict:class and value:float(P)): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = {}
        self.posteriors = []

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        # clear for refitting
        self.priors = {}
        self.posteriors = {}

        # set priors -- count the number of each class and divide by total instances
        freq_dict = {}
        for label in y_train:
            if label in freq_dict.keys():
                freq_dict[label] += 1
            else:
                freq_dict[label] = 1
        for key, value in freq_dict.items():
            self.priors[key] = value/len(y_train)

        # set posteriors
        self.posteriors = [{}] * len(X_train[0])

        for col_index in range(len(X_train[0])):
            self.posteriors[col_index] = {}

            # setup index's storage structure
            column = []
            for row_index in range(len(X_train)):
                column.append(X_train[row_index][col_index])
            item_label_list, parallel_frequency_list = utils.find_frequency_of_each_element_in_list(column)
            for item in item_label_list:
                self.posteriors[col_index][item] = {}
                y_label_list, y_parallel_frequency_list = utils.find_frequency_of_each_element_in_list(y_train)
                for y_label in y_label_list:
                    self.posteriors[col_index][item][y_label] = 0
                    
            # find number of occurances for each class based on a given label
            for row_index in range(len(X_train)):
                value_at_index = X_train[row_index][col_index]
                class_at_index = y_train[row_index]
                self.posteriors[col_index][value_at_index][class_at_index] += 1
            # divide each total
            for val_key in self.posteriors[col_index].keys():
                for class_key in self.posteriors[col_index][val_key].keys():
                    self.posteriors[col_index][val_key][class_key] /= (self.priors[class_key] * len(y_train))

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        class_labels = list(self.priors.keys())
        for test_item in X_test:
            best_value = 0
            best_index = 0
            for i, class_label in enumerate(class_labels):
                prior_val = self.priors[class_label]
                for j in range(len(test_item)):
                    if test_item[j] in self.posteriors[j]:
                        prior_val *= self.posteriors[j][test_item[j]][class_label]
                if best_value < prior_val:
                    best_value = prior_val
                    best_index = i
            y_predicted.append(class_labels[best_index])
                
        return y_predicted