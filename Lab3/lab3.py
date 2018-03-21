#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
from vegas_regression import vegas_regression
from vegas_classification import vegas_classification

string_dict = {}

# isInt
#   Parameters:
#       value   -   Object to check if Int
#   Purpose:
#       Return a boolean result telling whether the parameter is an int
def isInt(value):
    try:
        num = value + 1
        return True
    except TypeError:
        return False

# stringToIndex
#   Parameters:
#       value   -   The value to add or get the index of
#       key     -   The key to reference in string_dict for value
#   Purpose:
#       Convert a given value into a numerical index so tensorflow can use it.
def stringToIndex(value, key):
    if key not in string_dict.keys():
        string_dict[key] = []

    if value not in string_dict[key]:
        string_dict[key].append(value)
        return len(string_dict[key])

    for i,j in enumerate(string_dict[key]):
        if j == value:
            return i+1
    return -1


###################
#    Main Body    # 
###################
print('Written by Andrew Schaffer\n')

data = pd.read_csv('vegas2.csv')                            #Open the file and read data
ratings_data = []
n_features = len(data.keys()) - 1

for index, row in data.iterrows():                          #Store data in a matrix
    ratings_data.append({'Data':[], 'Score':0, 'Hotel name':0})
    for key in row.keys():                                  #Store row data in the ratings_data array
        if key == 'Score':                                  #Keep score in a separate index for the Y in vegas_regression
            ratings_data[-1]['Score'] = row[key]            
        elif key == 'Hotel name':                             #Keep name in separate index for the Y in vegas_classification
            ratings_data[-1]['Hotel name'] = stringToIndex(row[key], key)
        elif key == 'Nr. rooms':                              #Skip irrelevant information
            continue
        elif key == 'Review weekday':
            continue
        elif key == 'Member years':
            continue
        else:
            if isInt(row[key]):                             #Check if the value is a string
                ratings_data[-1]['Data'].append(row[key])
            else:
                intValue = stringToIndex(row[key], key)     #Convert the string to an int
                ratings_data[-1]['Data'].append(intValue)
    #End for loop
#End for loop

sess = tf.Session()                                         #Create a session to pass to the referenced classes
regression = vegas_regression(sess, ratings_data)

for i in range(0, 5):                                      #Get a number of predictions from the regression class
    regression.predict()
print('Lab Regression Question Predictions')
regression.labPredict()

#classificatsion = vegas_classification(sess, ratings_data)
#for i in range(0, 5):
#    classification.predict()

#Show loss history on graph
pid = os.fork()
if pid == 0:
    regression.show_loss()

sess.close()
