# -*- coding: utf-8 -*-
"""
@author: Brad Pershon
"""
from __future__ import division
import numpy as np

#Function returns a set of training samples, samples are drawn with replacement
#   pop_size: total number of samples
#   train_size: Size of the training set
#   arry: mdarray from which the samples are drawn
#   W: sample weights, the probability of being drawn
def get_train_set (train_size, arry, W):
    train_set_index = np.random.choice(train_size, train_size, p=W)   
    return arry[train_set_index.tolist(), :]

#Function calculates basic classification perforamce measures
#   tp: count of true postives
#   fn: count of false negatives
#   fp: count of false postives
#   tn: count of true negativess
def performance(tp, fn, fp, tn):
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    fmeas = 2*tp / (2*tp + fn + fp)
    print ""
    print "Results"
    print "------------------------------------------------------"
    print "Accuracy: ", acc
    print "Precision: ", prec
    print "Recall: ", recall
    print "F-Measure: ", fmeas
