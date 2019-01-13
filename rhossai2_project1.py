#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 23:43:06 2018

@author: rayhan
"""

import math 
import numpy as np


Y = []
X = []
XT = []
YT = []
std_X = []
std_Y = []
std_XT = []
std_YT = []

missing_values = []
Means = []
std_Dev = []


#import data and remove the missing values
with open('auto-mpg.data', 'r') as f:
    dataset = f.readlines()    
    for lines in dataset:
        values = lines.split()
        if(values[3] == '?'):
            missing_values.append(lines)
        else:
            Y.append(float(values[0]))
            X.append(list(map(float, values[1:8])))
  
#calculate the mean value          
def cal_mean(matrix, i):
    sums = 0
    
    for j in range(len(matrix)):
        sums = sums + matrix[j][i]
    return sums / len(matrix)

#replace missing values with column average
def fill_missing_values():
    missing_mean = cal_mean(X,2)
    for x in missing_values:
        values = x.split()
        Y.append(float(values[0]))
        values[3] = missing_mean
        X.append(list(map(float, values[1:8])))

#generate the transpose of a given matrix
def transpose(mat):
    return np.array(mat).T.tolist()
        
        
def standardization(do_std_y):
    column = 0
    if do_std_y == 0:
        
        if len(Means) == 0:
            while column < 7:
                means = cal_mean(X,column)
                column+=1
                Means.append(means)
        MT = transpose(X)
        for i, x in enumerate(MT):
            diff = np.array(x) - Means[i]        
            diff_sqr = [j * j for j in diff]
            d = math.sqrt( np.sum(diff_sqr) / len(diff_sqr))
            std_Dev.append(d)
            std_d = diff / d
            std_XT.append(std_d)
        return np.array(std_XT).T.tolist()
    else:
        mean_Y = np.sum(Y)/len(Y)
        YT = transpose(Y)
        diff=[]
        diff[:]= [x-mean_Y for x in YT]
        diff_sqr = [j * j for j in diff]
        d = math.sqrt( np.sum(diff_sqr) / len(diff_sqr))
        std_d=[]
        std_d[:]= [x/d for x in diff]
        std_YT.append(std_d)
        return np.array(std_YT).T.tolist()
        
    


def linear_regression(matrix):
    matrix_X = np.insert(matrix, 0, 1, axis = 1)
    matrix_XT = matrix_X.transpose()
    
    XTX = np.matmul(matrix_XT, matrix_X)
    XTXinverse = np.linalg.inv(XTX)
    Xprod = np.matmul(XTXinverse, matrix_XT)
    w = np.matmul(Xprod, Y_train)
    return w


#start of the main function
fill_missing_values()

XT = transpose(X)
YT = transpose(Y)

std_X = standardization(0)
std_Y = standardization(1)


X_train = std_X[:int(len(std_X)*.7)]
Y_train = std_Y[:int(len(std_Y)*.7)]
X_test = std_X[int(len(std_X)*.3):]
Y_test = std_Y[int(len(std_Y)*.3):]

linear_result = linear_regression(X_train)


linear_result_transpose = transpose(linear_result[:-1])
X_test_transpose = transpose(X_test)

Y_predict = np.matmul(linear_result_transpose, X_test_transpose)
Y_predict_transpose = transpose(Y_predict)

#print and see the Y_predict
print(Y_predict_transpose)
