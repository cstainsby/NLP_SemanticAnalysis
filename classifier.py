
import math
import numpy as np
import utils
from collections import Counter, defaultdict


# from mysklearn import myevaluation, myutils
# from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
# import mysklearn


def train_naive_bayes(bag_of_words_pos, bag_of_words_neg):
  """trains the data by calculating priors and likelhoods 
        Args:
            bag_of_words_pos(list): list of positive words
            bag_of_words_neg(list): list of negative words
        Returns:
            priors(dict): dictionary of priors
            likelihoods(nested dict): nested dictionary of likelihoods
    """
  priors = {
    0: 0.5,
    1: 0.5
  }

  likelihoods = {}

  # adding in likelihoods for positive words
  unique_words = list(set(bag_of_words_pos))
  for unique_word in unique_words:
    likelihoods[unique_word] = {}

     # calculating likelihood percentage for negative word
    count_of_unique_word = bag_of_words_pos.count(unique_word)
    likelihoods[unique_word][1] = count_of_unique_word / len(bag_of_words_pos)
  
  # adding in likelihoods for negative words
  unique_words = list(set(bag_of_words_neg))
  for unique_word in unique_words:
    
    # word was not positive
    if unique_word not in likelihoods:
      likelihoods[unique_word] = {}
    
    # calculating likelihood percentage for negative word
    count_of_unique_word = bag_of_words_neg.count(unique_word)
    likelihoods[unique_word][0] = count_of_unique_word / len(bag_of_words_neg)

  # likelihood example: {time: {0: 0.02} {1: 0.03}}
  return priors, likelihoods

def predict_naive_bayes(priors, likelihoods, test):
  predictions = []

  class_labels = list(priors.keys())
  for test_item in test:
    
    sum = [0 for i in range(len(priors))]
    for i, class_label in enumerate(class_labels):
      sum[i] = priors[class_label]
      for word in test_item:
        if word in likelihoods: 
          if class_label in likelihoods[word]:
              sum[class_label] += likelihoods[word][class_label]

    predictions.append(sum.index(max(sum)))
          
  return predictions