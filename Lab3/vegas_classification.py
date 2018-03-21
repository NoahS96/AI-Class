#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Luis Sanchez
@created: 3/8/18

Description:
    This file should contain a function for predicting the best hotel
    based upon certain parameters. The result must be one of the hotel names in the
    provided dataset   

Tasks:
    Use parameters to determine the best hotel
    Return the best hotel name
"""
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

global sess
global features, labels

# Parse data from dataset
def load_vegas():
    #dataset = pd.read_csv('vegas.csv', usecols=lambda x: x.upper() in ['NR. REVIEWS', 'NR. HOTEL REVIEWS','HELPFUL VOTES', 'PERIOD OF STAY', 'TRAVELER TYPE','HOTEL NAME', 'HOTEL STARS'])
    dataset = pd.read_csv('vegas.csv', usecols=lambda x: x.upper() in ['PERIOD OF STAY', 'TRAVELER TYPE', 'HOTEL STARS', \
                                                                       'POOL', 'GYM', 'TENNIS COURT', 'SPA', 'CASINO', 'FREE INTERNET'])
    
    # Data modeling to categories
    dataset['Period of stay'] = dataset['Period of stay'].astype('category').cat.codes
    dataset['Traveler type'] = dataset['Traveler type'].astype('category').cat.codes
    dataset['Pool'] = dataset['Traveler type'].astype('category').cat.codes
    dataset['Gym'] = dataset['Traveler type'].astype('category').cat.codes
    dataset['Tennis court'] = dataset['Traveler type'].astype('category').cat.codes
    dataset['Spa'] = dataset['Traveler type'].astype('category').cat.codes
    dataset['Casino'] = dataset['Traveler type'].astype('category').cat.codes
    dataset['Free internet'] = dataset['Traveler type'].astype('category').cat.codes
    
    
    
    # Set Labels
    target = pd.read_csv('vegas.csv', usecols=lambda x: x.upper() in ['HOTEL NAME'])
    target['Hotel name'] = target['Hotel name'].astype('category').cat.codes
    
    # return features to input tensors
    return dataset.as_matrix(), target.as_matrix()



def c_train():
    global sess
    global features, labels
    global X, Y, y_
    # get features and labels from vegas dataset
    features, labels = load_vegas()

    # how many columns are there in features
    n_dim = features.shape[1]
    
    # Divide data into training and testing sets
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.33, random_state=42)
    
    # Set up hyper paramethers
    learning_rate = 0.0001
    training_epochs = 5000
    display_step = 1000
    
    # Create computation 
    X = tf.placeholder(tf.float32,[None,n_dim])
    Y = tf.placeholder(tf.float32, [None, 1])
    W = tf.Variable(tf.zeros([n_dim, 1]))
    
    # Our learning argrithm, classifies base on cost
    y_ = tf.matmul(X, W)
    cost = tf.reduce_mean(tf.square(y_ - Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    # Initialize all declred variables
    init = tf.global_variables_initializer()
    
    # initialize tensorflow session
    sess = tf.Session()
    sess.run(init)
    
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={X: train_x, Y:train_y})
        
        # Print results as minimized the cost function
        if (epoch) % display_step == 0:
            cc = sess.run(cost, feed_dict={X: train_x, Y:train_y})
            print('Training step:', "%04d" % (epoch), 'cost=', "{:f}".format(cc))

    print("Optimization Finished!")
    #training_cost = sess.run(cost, feed_dict={X: train_x, Y:train_y})
    #w_ = sess.run(W)
    #print("Training cost=", training_cost, "W=", w_)
    
    



def c_predictHotel(period="Dec-Feb", tType="Solo"):
    global sess, y_, X
    dataset = pd.read_csv('vegas.csv', usecols=lambda x: x.upper() in ['PERIOD OF STAY', 'TRAVELER TYPE', 'HOTEL NAME',\
                                                                       'POOL', 'GYM', 'TENNIS COURT', 'SPA', 'CASINO', 'FREE INTERNET'])
    period_ = dataset['Period of stay'] = dataset['Period of stay'].astype('category')
    tType_ = dataset['Traveler type'] = dataset['Traveler type'].astype('category')
    hotels_ = dataset['Hotel name'] = dataset['Hotel name'].astype('category')
    
    # default values
    stars = 5
    
    # Amenities
    pool = 0
    gym = 0
    tennis = 0
    spa = 0
    casino = 0
    internet = 0
    
    # Get index of period to use for prediction
    pi = 0
    if period == "Dec" or \
       period == "Jan" or \
       period == "Feb":
        pi = 0
    elif period == "Mar" or \
         period == "Apr" or \
         period == "May":
        pi = 2
    elif period == "Jun" or \
         period == "Jul" or \
         period == "Aug":
        pi = 1
    elif period == "Sep" or \
         period == "Oct" or \
         period == "Nov":
        pi = 3
    
    
    # Get index of traveler type to use for prediction
    # Depending on the traveler type its the ameneties choosen
    tt = 0
    if tType == "Solo":
        tt = 4
        gym = 1
        tennis = 1
        casino = 1
        spa = 1
        internet = 1
    elif tType == "Friends":
        tt = 3
        pool = 1
        casino = 1
        internet = 1
    elif tType == "Families":
        tt = 2
        pool = 1
        casino = 1
    elif tType == "Couples":
        tt = 1
        spa = 1
        casino = 1
    elif tType == "Business":
        tt = 0
        internet = 1
        

    # Used parameters given to function to for a prediction
    predictFrom = np.array([[pi, tt, stars, pool, gym, tennis, spa, casino, internet]])
    
    pred_own = sess.run(y_, feed_dict={X: predictFrom})
    
    # round up or down prediction
    prediction = round(pred_own.item(0))
    
    # Using the index gotten get the name of the hotel
    print("For a", tType, "trip in", period , "I recommend the:", hotels_.cat.categories[prediction])
    
    


def c_close():
    global sess
    sess.close()

