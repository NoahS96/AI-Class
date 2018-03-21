#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class vegas_regression:

    tf_sess = None
    dataset = None
    loss_hist = np.empty(shape=[1], dtype=float)
    y_ = None
    X = None
    Y = None
    W = None
    b = None

    # __init__
    #   Parameters:
    #       tf_sess -   The tensorflow session to use
    #       dataset -   The dataset containing ratings data and the corresponding scores
    #   Purpose:
    #       Save and normalize the provided dataset and set up the linear regression varialbes
    #       in tensorflow. Train the tensorflow model with the provided dataset.
    def __init__(self, tf_sess, dataset):
        self.dataset = dataset

        #Normalize the dataset and append the hotel name to the data array
        for i,j in enumerate(self.dataset):
            self.dataset[i]['Data'].append(self.dataset[i]['Hotel name'])
            self.dataset[i]['Data'] = self.normalize(self.dataset[i]['Data'])

        #Get shapes
        x_temp = tf.constant(self.dataset[0]['Data'])
        y_temp = tf.constant(self.dataset[0]['Score'])
        x_shape = x_temp.get_shape()
        y_shape = y_temp.get_shape()
        n_features = len(self.dataset[0]['Data'])
      
        #Parameters
        learning_rate = 0.1
        
        #Set placeholders 
        self.X = tf.placeholder(tf.float32, shape=x_shape, name="x_in")
        self.Y = tf.placeholder(tf.float32, shape=y_shape, name="y_in")
        self.W = tf.Variable(tf.zeros([1,n_features]), name="w")
        self.b = tf.Variable(np.random.randn(), name="bias")
        init = tf.global_variables_initializer()
         
        #Calculate result and loss
        self.y_ = tf.add(tf.multiply(self.X, self.W), self.b)
        loss = tf.reduce_mean(tf.square(self.y_ - self.Y))
        train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        
        self.tf_sess = tf_sess
        self.tf_sess.run(init)

        #Train the model and save the loss history
        for i in range(0, 10):
            np.random.shuffle(self.dataset)
            for rating in self.dataset:
                self.tf_sess.run(train, feed_dict={self.X:rating['Data'], self.Y:rating['Score']})
                self.loss_hist = np.append(self.loss_hist, self.tf_sess.run(loss, feed_dict={self.X:rating['Data'], self.Y:rating['Score']}))

    # show_loss
    #   Purpose:
    #       Display the loss history using matplotlib
    def show_loss(self):
        plt.plot(range(len(self.loss_hist)), self.loss_hist)
        plt.axis([0, len(self.loss_hist), 0, np.max(self.loss_hist)])
        plt.show()

    # labPredict
    #   Purpose:
    #       Make a prediction based on the lab parameters
    def labPredict(self):
        #Martin family in circus circus for October
        test1 = [1,11,4,13,4,3,1,2,1,1,2,2,1,3,1,11]

        #Jackson on business trip staying in Bellagio during January
        test2 = [2,119,7,3,1,2,2,2,1,2,2,2,17,5,2,1]

        pred_y1 = self.tf_sess.run(self.y_, feed_dict={self.X:test1})
        pred_y2 = self.tf_sess.run(self.y_, feed_dict={self.X:test2})
        print('Prediction for Martin family: %.2f' % (self.tf_sess.run(tf.reduce_mean(pred_y1))))
        print('Prediction for Jackson: %.2f' % (self.tf_sess.run(tf.reduce_mean(pred_y2))))

    # predict
    #   Purpose:
    #       Get a random index from the dataset and pass the data to the tensor object to get a prediction.
    #       Compare the result to the actual score in the dataset and calculate the MSE.
    def predict(self):
        rand_index = np.random.randint(0,high=(len(self.dataset)-1))
        pred_x = self.dataset[rand_index]['Data']
        pred_y = self.dataset[rand_index]['Score']
        test_x = self.dataset[~rand_index]['Data']
        test_y = self.dataset[~rand_index]['Score']
        pred_y = self.tf_sess.run(self.y_, feed_dict={self.X:pred_x})
        mse = tf.reduce_mean(tf.square(pred_y - test_y))
        print('predicted: %.2f' % (self.tf_sess.run(tf.reduce_mean(pred_y))))
        print('actual: %.2f' % (test_y))
        print('MSE: %.2f\n' % (self.tf_sess.run(mse)))

    # normalize
    #   Parameters:
    #       data    -   The data array to normalize
    #   Purpose:
    #       Convert the array values to a normalized float value
    def normalize(self, data):
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return (data-mu)/sigma


