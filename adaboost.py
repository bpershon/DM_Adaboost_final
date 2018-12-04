# -*- coding: utf-8 -*-
"""
@author: Brad Pershon
"""

from __future__ import division
import numpy as np

class AdaBoost:
    
    def __init__(self, training_size):
        self.N = training_size
        self.weights = np.ones(self.N)/self.N
        self.CLASSIFIERS = []
        self.ALPHA = []
        
    def set_training_set(self, training_set):
        self.training_set = training_set

    def test_classifier(self, classifier):
        #Generate an array of booleaon values, checks if the predicted value is equal to the ground truth for the training set
        errors = np.array([t[0]!=classifier.predict(t[1:]) for t in self.training_set])
        #Calculate the weighted error
        e = (errors*self.weights).sum() / self.N
        #If the error is to high, we need to train another classifier
        if e > 0.5:
            return True
        #Otherwise calc alpha, update weights, add classifier to our ensemble
        alpha = 0.5 * np.log((1-e)/e) 
        w = np.zeros(self.N)
        for i in range(self.N):
            #Update the weight of each example
            if errors[i] == 1: 
                #Error was found, increase weight
                w[i] = self.weights[i] * np.exp(alpha)
            else: 
                w[i] = self.weights[i] * np.exp(-alpha)
        self.weights = w / w.sum() 
        self.CLASSIFIERS.append(classifier)
        self.ALPHA.append(alpha)
        return False

    def predict(self, test_set):
        num_classifiers = len(self.CLASSIFIERS)
        tp, tn, fp, fn = 0, 0, 0, 0
        for x in test_set:
            #Generate a list of predictions for each class record
            predicted_list = [self.ALPHA[i]*self.CLASSIFIERS[i].predict(x[1:]) for i in range(num_classifiers)]
            #Ground Truth
            actual = np.sign(x[0])
            #Sum the predictions from the Ensemble and choose class based on its sign (class strength)
            predicted = np.sign(sum(predicted_list))
            #Assign entry to Confusion matrix
            if actual == 1.0:
                if predicted == 1.0:
                    tp += 1
                else:
                    fn += 1
            else:
                if predicted == -1.0:
                    tn += 1
                else:
                    fp += 1
        return [tp, fn, fp, tn]
                        