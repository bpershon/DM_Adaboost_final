# -*- coding: utf-8 -*-
"""
@author: Brad Pershon
Data Mining
Final Project
AdaBoost
Classification pg 185
"""

from __future__ import division
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import math
import util
import adaboost
pd.options.mode.chained_assignment = None #default='warn'
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

#Determiines the % of records in the test / training sets
training_size_perc = 0.7
#KNN k value
knn_k = 7;
#Import Data
inputFile = "data.csv"
df = pd.read_csv(inputFile, header = 0)

#PREPROCESSING 
del df['id']
# -1 = M and 1 = B
df_class = df['diagnosis']
del df['diagnosis']
#Normalize data
df = (df - df.min()) / (df.max() - df.min())
#Reinsert the ground truth labels
df.insert(loc=0, column='ground_truth', value = df_class)

#Main
if __name__ == '__main__':
    k = int(raw_input("Please enter the number of boosting iterations: "))
    if k == 0: 
        k = 5
    print "Iteration value: ", k
    classifierType = int(raw_input("Choose base classifier 1 - Decision Trees or 2 - KNN: "))
    if classifierType != 1 and classifierType != 2: 
        classifierType = 1
    print "Classifier choice: ", classifierType
    
    #Split into training and test sets, Select training records
    train_size = int(math.floor(training_size_perc * len(df)))
    df_size = len(df)
    df_array = df.as_matrix()
    np.random.shuffle(df_array)
    training_master, test_set = df_array[:train_size, :], df_array[train_size+1:, :]
    #Init Ensemble Learner
    learner = adaboost.AdaBoost(train_size);
    #Train K base classifiers for the Ensemble learner 
    for i in range(k):
        #Generate training set with replacement
        training_set = util.get_train_set(learner.N, df_array, learner.weights)
        learner.set_training_set(training_set)
        #Choose base classifier
        if classifierType == 1:
            classifier = tree.DecisionTreeClassifier()
        else:
            classifier = KNeighborsClassifier(n_neighbors=knn_k)
        #Train base classifier
        classifier = classifier.fit(learner.training_set[:, 1:], learner.training_set[:, 0])
        #Apply baes clasifier to training_master, verify error is less than 0.5, if so add classifier and update weights
        learner.set_training_set(training_master)
        test = learner.test_classifier(classifier)
        #If the classifier failed the test, redraw the samples and rerun this iteration
        if(test):
            i -= 1
            continue
        
    #Test Adaboost with test set
    [tp, fn, fp, tn] = learner.predict(test_set)
    #Output performance measures
    util.performance(tp, fn, fp, tn)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        