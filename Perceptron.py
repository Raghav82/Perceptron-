import pandas as pd 
import numpy as np
import random
from random import randrange

# DATA section.  The following is to get and prepare a dataset.  Dataset is 208 rows with 61 columns.
# Sonar observations recorded as decimal values < 1. Column 61 is labeled Marine or Rock which we change into binary below
'''
dataset=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data",header=None)
dataset[dataset.shape[1]-1]=pd.Categorical(dataset[dataset.shape[1]-1]).codes
dataset= np.asarray(dataset)
dataset = dataset.tolist()
'''
#Perceptron Code

def predict(row, bias_weights):

  bias = bias_weights[0]
  weights = bias_weights[1:]
  inputs = row[0:-1]
  output = np.dot(weights,inputs) + bias
  if output >= 0:
    return 1
  else:
    return 0
  
def train_weights(train , learning_rate,  epochs):
  bias_weights = np.zeros(len(train[0]))
  #bias_weights list is set to all zeroes initially
  for epoch in range(epochs):
    total_error = 0
    for row in train:
      prediction = predict(row,bias_weights)
      error = row[-1] - prediction
      total_error = total_error + error ** 2
      inputs = np.asarray(row[0:-1]) 
      bias_weights[0] = bias_weights[0] + (learning_rate * error)
      bias_weights[1:] =bias_weights[1:] + inputs * (learning_rate * error)
  return bias_weights




