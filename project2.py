
import time
import math
import copy
import csv
import numpy as np
import pandas as pd

def forwardSelection(inf, featureRange):
    start = time.perf_counter()
    bestFeatures = set()
    hashMap = {}

    for i in range(1, featureRange):
        bestAccuracy = 0
        jIndex = 0
        for j in range(1, featureRange):
            if j not in bestFeatures:
                tempFeatures = copy.copy(bestFeatures)

                # Add the feature to temporary set
                tempFeatures.add(j)

                # Reading in file into dataframe:
                dataFrame = pd.read_fwf(inf, header=None)

                dataFrameCopy = dataFrame.copy()[:-1]
                accuracy = kFold(featureRange, tempFeatures, dataFrameCopy)
                tempJ = j
                print('Using the set of feature(s) ' + str(tempFeatures) + ' accuracy is ' + "{:.2%}".format(accuracy))

                # Update the best accuracy 
                if accuracy >= bestAccuracy:
                    bestAccuracy = accuracy
                    finalAccuracy = accuracy
                    jIndex = tempJ

        # Add best feature to the set
        bestFeatures.add(jIndex)
        featureSetCopy = copy.copy(bestFeatures)
        hashMap[finalAccuracy] = featureSetCopy
        
        print('The feature set ' + str(bestFeatures) + ' had the best accuracy ' + "{:.2%}".format(finalAccuracy) + '\n')
        print('Anytime search result, the best feature subset is ' + str(hashMap[max(hashMap.keys())]) +
          ' with an accuracy of of ' + "{:.2%}".format(max(hashMap.keys())) + '\n')
        print('Time taken: ' + str(round(time.perf_counter()-start, 2)) + ' s.')

    print('Completed search! The best feature subset is ' + str(hashMap[max(hashMap.keys())]) +
          ' with an accuracy of of ' + "{:.2%}".format(max(hashMap.keys())) + '\n')
    print('Time taken so far: ' + str(round(time.perf_counter()-start, 2)) + ' s.')


def backwardElimination(inf, featureRange):
    start = time.perf_counter()
    bestFeatures = set()
    for j in range(1, featureRange):
        bestFeatures.add(j)
    hashMap = {}

    tempFeatures = copy.copy(bestFeatures)
    dataFrame = pd.read_fwf(inf, header=None)

    dataFrameCopy = dataFrame.copy(deep=True)

    accuracy = kFold(featureRange, tempFeatures, dataFrameCopy)
    print('Using feature(s) ' + str(tempFeatures) + ' accuracy is ' + "{:.1%}".format(accuracy))
    print('The feature set ' + str(tempFeatures) + ' was the best, accuracy is ' + "{:.1%}".format(accuracy) + '\n')

    for i in range(2, featureRange):
        bestAccuracy = 0
        jIndex = 0

        for j in range(1, featureRange):
            if j in bestFeatures:
                tempFeatures = copy.copy(bestFeatures)
                # Temporarily remove from the set
                tempFeatures.remove(j)

                # Reading the file into pandas dataframe
                dataFrame = pd.read_fwf(inf, header=None)

                dataFrameCopy = dataFrame.copy(deep=True)

                accuracy = kFold(featureRange, tempFeatures, dataFrameCopy)
                tempJ = j
                print('Using the set of feature(s) ' + str(tempFeatures) + ' accuracy is ' + "{:.1%}".format(accuracy))

                # Update the best accuracy 
                if accuracy >= bestAccuracy:
                    bestAccuracy = accuracy
                    finalAccuracy = accuracy
                    jIndex=tempJ

        # Remove the feature from set
        bestFeatures.remove(jIndex)
        featureSetCopy = copy.copy(bestFeatures)
        hashMap[finalAccuracy] = featureSetCopy

        print('The feature set ' + str(bestFeatures) + ' had the best accuracy ' + "{:.2%}".format(finalAccuracy) + '\n')
        print('Anytime search result, the best feature subset so far is ' + str(hashMap[max(hashMap.keys())]) +
          ' with an accuracy of ' + "{:.2%}".format(max(hashMap.keys())) + '\n')
        print('Time taken so far: ' + str(round(time.perf_counter()-start, 2)) + ' s.\n')
    
    print('Completed search! The best feature subset is ' + str(hashMap[max(hashMap.keys())]) +
          ' with an accuracy of of ' + "{:.2%}".format(max(hashMap.keys())) + '\n')
    print('Time taken: ' + str(round(time.perf_counter()-start, 2)) + ' s.')


def kFold(featureRange, currentFeatures, dataFrameCopy):
    noOfRows = len(dataFrameCopy.index)
    correctlyClassified = 0

    dataFrame = dataFrameCopy.copy()

    
    
    dataFrameCopy = dataFrame.to_numpy()

    # Set features not used to 0
    for i in range(1, featureRange):
        if i not in currentFeatures:
            dataFrameCopy[:, i] = 0.0

    # Perform nearest neighbor classification
    for j in range(noOfRows):
        classifiedObjects = dataFrameCopy[j][1:featureRange]
        classifiedObjectsLabels = dataFrameCopy[j][0]

        nearestNDistance = float('inf')
        nearestNIndex = float('inf')

        for k in range(noOfRows):
            dist = 0

            # Compute distance and update 
            if j != k:
                dist = math.sqrt(np.sum(np.power(classifiedObjects - dataFrameCopy[k][1:featureRange], 2)))

                if dist <= nearestNDistance:
                    nearestNDistance = dist
                    nearestNIndex = k + 1
                    nearestNLabel = dataFrameCopy[nearestNIndex - 1][0]

        # If classified correctly, increment the counter
        if classifiedObjectsLabels == nearestNLabel:
            correctlyClassified += 1
    accuracy = correctlyClassified/noOfRows
    return accuracy

def menu():
    # Choosing a data set size
    choice = int(input('Enter the number for the preferred dataset size:\n'
                     '\n1. Small'
                     '\n2. Large\n\n'))
    if choice == 1:
        file = "CS205_SP_2022_SMALLtestdata__57.txt"
    elif choice == 2:
        file = "CS205_SP_2022_Largetestdata__59.txt"
    else: 
        print('Terminating due to incorrect input.') 

    fp = file
    file = open(file, 'r')

    row = csv.reader(file, delimiter=' ', skipinitialspace=True)
    noOfFeatures = len(next(row))

    choice = int(input('Enter the number for the preferred algorithm:\n'
                     '\n1. Foward Selection'
                     '\n2. Backward Elimination\n\n'))

    # Choosing a feature selection technique
    if choice == 1:
        return forwardSelection(fp, noOfFeatures)
    elif choice == 2:
        return backwardElimination(fp, noOfFeatures)
    else: 
        print('Terminating due to incorrect input.') 


menu()
