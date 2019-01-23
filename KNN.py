import sys
import pandas as pd
import numpy as np
import math
import operator

#ensures correct number of arguments
if not(3 <= len(sys.argv) <= 4) :
    print("Invalid number of arguments.\n")
    exit()

def euclid_distance(point1,point2, length):
    '''returns the euclid_distance between point1 and point2'''
    distance = 0
    for x in range(length):
        distance += np.square(point1[x]-point2[x])
    return np.sqrt(distance)

def knn(trainingSet, testSet, k) :
    '''calculates knn'''
    distances ={}
    sort ={}

    length = testSet.shape[1]

     # Calculating euclidean distance between each row of training data and test data
    for x in range(len(trainingSet)):
        dist = euclid_distance(testSet, trainingSet.iloc[x], length)
        distances[x] = dist[0]

    # Sorting them on the basis of distance
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))

    neighbors = []
    #top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])

    classVotes = {}

     # Calculating the most frequent class in the neighbors
    for x in range(len(neighbors)):
        freq = trainingSet.iloc[neighbors[x]][-1]

        if freq in classVotes:
            classVotes[freq] += 1
        else:
            classVotes[freq] = 1

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    #to calculate conditional probability
    #need the class votes of the neighbours
    neighbourClasses = np.asarray(sortedVotes)
    neighbourClasses =neighbourClasses.ravel()

    return(int(sortedVotes[0][0]), neighbourClasses)


def get_accuracy(testSet, prediction):
    '''calculates conditonally probability of prediction and its neighbors'''
    correct = 0
    for x in range(len(testSet)):
        if testSet[x] == prediction:
            correct += 1
    return correct/float(len(testSet))

def main():

    trainingSet = pd.read_csv(sys.argv[1], sep="\t")
    testSet=pd.read_csv(sys.argv[2], sep="\t")
    testSet.columns = ['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']

    #default value for k if not supplied
    try:
        sys.argv.append('')
        k = int(sys.argv[3])
    except ValueError:
        k = 3
    else :
        k = int(sys.argv[3])

    predictions = []
    for i, row in testSet.iterrows():
        eachTestcase = pd.DataFrame([[row.RI,row.Na,row.Mg,row.Al,row.Si,row.K,row.Ca,row.Ba,row.Fe]])
        result, neighbourClasses = knn(trainingSet,eachTestcase,k)
        prob = get_accuracy(neighbourClasses,result)
        print(repr(result) +"  "+ repr(prob))
main()
