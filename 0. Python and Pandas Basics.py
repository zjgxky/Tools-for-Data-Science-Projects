#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 19:02:29 2021

@author: Kaiyan Xu
homework-0-zjgxky
"""

import random
import numpy as np

# Q1
def import_data(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    X = []
    y = []
    for line in lines:                                  
        line = line.strip()
        difwords = line.split(',')
        for index in range(len(difwords)):
            if difwords[index] == '?':
                # replace '?' with nan
                difwords[index] = np.nan
        
        Xwords = difwords[0:279]
        ywords = difwords[279]
            
        X.append(Xwords)
        y.append(ywords)
        
    return X,y

# Q2(a)
# Q2(a) helper function: compute the median of the column
def median(line):
    column = []
    for num in range(279):        
        if np.isnan(line[num]) == False:
            column.append(line[num])
    column.sort()
    index = len(column) // 2 - 1
    # the length is even
    if len(column) % 2 == 0:
        med = (column[index] + column[index + 1]) / 2
    else:
        # the length is odd
        med = column[index]
    return med

# Q2(a)
def impute_missing(X): 
    # transpose X
    firsttrans =[[row[i] for row in X] for i in range(len(X[0]))]
    for line in firsttrans:
        final = [float(number) for number in line]
        med = median(final)
        for num in range(279):
            if np.isnan(final[num]):
                line[num] = med
    # transpose back
    transback = [[row[i] for row in firsttrans] for i in range(len(firsttrans[0]))]
    return transback

# Q2(b)
"""
Some outliers largely influences mean, which makes the mean not representative to
most datas. Median is more representative.
"""

# Q2(c)
def discard_missing( X, y ):
    newX = []
    newy = []
    numLine = -1
    for line in X:
        containNAN = False
        numLine = numLine + 1
        for number in range(279):
            final = [float(number) for number in line]
            if np.isnan(final[number]):
                containNAN = True
                break
        # does not contain nan, so add in new lists
        if containNAN == False:
            newX.append(line)
            newy.append(y[numLine])
    return newX, newy

# Q3(a)
def shuffle_data ( X, y ):
    # shuffle with the same order
    temp = list(zip(X, y))
    random.shuffle(temp)
    shuffleX, shuffley = zip(*temp)
    return shuffleX, shuffley

# Q3(b) helper function: return mean of the column
def compute_mean(X, index):
    sum = 0
    for line in X:
        # turn into float to add later
        final = [float(number) for number in line]
        sum = sum + final[index]
    mean = sum/len(X)
    return mean

# Q3(b)
def compute_std( X ): 
    std = []
    for index in range(279):
        sumsq = 0
        mean = compute_mean(X, index)
        for line in X:
            # turn into float for calculation later
            final = [float(number) for number in line]
            sumsq = sumsq + (final[index] - mean) ** 2
        dev = sumsq/(len(X) - 1)
        onestd = dev ** (1/2)
        std.append(onestd)        
    return std

# Q3(c) 
def remove_outlier ( X, y ): 
    std = compute_std(X)
    mean = []
    # create a list for mean in each column
    for singleMean in range(279):
        mean.append(compute_mean(X, singleMean))
    for number in range(279):
        upperbound = mean[number] + 2*std[number]
        lowerbound = mean[number] - 2*std[number]
        for line in X:
            if line != []:
                # turn into float for calculation later
                floatnum = float(line[number])
                if floatnum > upperbound or floatnum < lowerbound:
                    line[number] = np.nan
    discard = discard_missing( X, y )                
    return discard[0], discard[1]

# Q3(d)
def standardize_data ( X ):
    newX = []
    std = compute_std(X)
    mean = []
    for singleMean in range(279):
        mean.append(compute_mean(X, singleMean))
    for line in X:
        final = [float(number) for number in line]
        for number in range(279):
            # avoid the situation when divisor equals to 0
            if std[number] != 0:
                final[number] = (final[number] - mean[number])/std[number]
        newX.append(final)
    return newX

# Q3(d)
""" 
Time efficiency is O(n*m) where n is the length of the 2-D list and m is 
the length of each list, because it iterates each line of the 2-D list once, 
and in each line, it iterates each element once. Therefore, O(n)*O(m) = O(n*m)
because n does not equal to m, so it is not O(n^2) in this case.
Space efficiency is also O(n*m) because each element, which is a single list, 
will be referred by each position in the newX[] list, and inside each list, 
there are m elements. Therefore, since it is two dimensional, O(n)*O(m) = O(n*m). 
"""

# Q4
def import_Titanicdata(filename):
    with open(filename, "r") as f:
        next(f)
        X = []
        y = []
        for line in f:                            

            line = line.strip()
            # using corresponding numbers to represent the data
            line = line.replace('female', '0')
            line = line.replace('male', '1')
            line = line.replace('C', '0')
            line = line.replace('Q', '1')
            line = line.replace('S', '2')
            difwords = line.split(',')
            for index in range(len(difwords)):
                if difwords[index] == '':
                    # use nan to show there is no data in this point
                    difwords[index] = np.nan
            difwords.pop(3)
            difwords.pop(3)
            difwords.pop(7)
            difwords.pop(8)
                        
            ywords = difwords.pop(1)
            Xwords = difwords
            
            X.append(Xwords)
            y.append(ywords)
        
    return X,y

# Q5(a)
def train_test_split( X, y, t_f ): 
    temp = list(zip(X, y))
    random.shuffle(temp)
    shuffleX, shuffley = zip(*temp)
    # number spliting to test sets
    splitnum = int(t_f * len(X))
    X_test = shuffleX[:splitnum]
    X_train = shuffleX[splitnum:]
    y_test = shuffley[:splitnum]
    y_train = shuffley[splitnum:]
    return X_train, y_train, X_test, y_test

# Q5(b)
def train_test_CV_split(X, y, t_f, cv_f): 
    temp = list(zip(X, y))
    random.shuffle(temp)
    shuffleX, shuffley = zip(*temp)
    # number spliting to test sets
    splitnum = int(t_f * len(X))
    # number spliting to cross-validation sets
    splitnum2 = int(cv_f * len(X))
    X_test = shuffleX[:splitnum]
    X_cv = shuffleX[splitnum : splitnum+splitnum2]
    X_train = shuffleX[splitnum+splitnum2:]
    y_test = shuffley[:splitnum]
    y_cv = shuffley[splitnum : splitnum+splitnum2]
    y_train = shuffley[splitnum+splitnum2:]
    return X_train, y_train, X_test, y_test, X_cv, y_cv
