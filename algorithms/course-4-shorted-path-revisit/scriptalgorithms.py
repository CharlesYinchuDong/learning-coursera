### This covers all tasks for course 4
### Charles Yinchu Dong
### 03/09/2019

### Import packages
import numpy as np
import re
import random
import copy
from collections import Counter
import sys, threading
import heapq
import matplotlib.pyplot as plt

def week1():
    """
    Code up floyd warshall algorithms
    """
    ### Initialization
    fileLocation = 'week-1/g1.txt'
#    fileLocation = 'week-1/g2.txt'
#    fileLocation = 'week-1/g3.txt'

    ### Load data
    with open(fileLocation, 'r') as f:
        dataRaw = f.read().splitlines()
    dataV1 = [x.split() for x in dataRaw[1:]]
    dataV2 = [[int(y) for y in x] for x in dataV1]
    numV = int(dataRaw[0].split()[0])
    numE = int(dataRaw[0].split()[1])
    
    ### Invoke Floyd Warshall
    lenPath, negCyc = floydWarshall(numV, numE, dataV2)

    if negCyc:
        print('There\'s negative cycle')
    else:
        print(lenPath)

    
#    print(dataRaw)
#    print(dataV2)
#    print(numV)
#    print(numE)

def floydWarshall(numV, numE, data):
    """
    Floyd Warshall APSP algorithms
    """
    ### Init
    A = np.full((numV, numV, 2), np.inf)
    minLen = float('inf') 
    existNegCyc = False
    ### Fill in the base case
    for edge in data:
        tail = edge[0] - 1
        head = edge[1] - 1
        edgeLen = edge[2]
        A[tail, head, 0] = edgeLen
        minLen = min(minLen, edgeLen)
    for i in range(numV):
        A[i, i, 0] = 0
    ### Dynamic programming
    for k in range(numV):
        for i in range(numV):
            for j in range(numV):
                case1 = A[i,j,0]
                case2 = A[i,k,0] + A[k,j,0]
                A[i,j,1] = min(case1, case2)
                if i != j:
                    minLen = min(minLen, A[i,j,1])
        negCyc = any([A[x, x, 1] != 0 for x in range(numV)])
        existNegCyc = any([existNegCyc, negCyc])
        if existNegCyc:
            return 0, True
        A[:,:,0] = A[:,:,1]
     
    return minLen, existNegCyc

def week2():
    """
    Code up Traveling Salesman Problem for week 2
    """
    ### Initialization
    fileLocation = 'week-2/tsp.txt'
#    fileLocation = 'week-2/tsp-v1.txt'

    ### Load data
    with open(fileLocation, 'r') as f:
        dataRaw = f.read().splitlines()
    numCity = int(dataRaw[0])
    dataV1 = [x.split() for x in dataRaw[1:]]
    dataV2 = [[float(y) for y in x] for x in dataV1]

#    print(dataV2)
#    plotCities(dataV2)    
    minDist1 = tsp(13, dataV2[:13])
    minDist2 = tsp(14, dataV2[11:])
    a1 = dataV2[11]
    a2 = dataV2[12]
    overlap = np.sqrt(((a1[0] - a2[0]) ** 2) + ((a1[1] - a2[1]) ** 2))

    print(minDist1)
    print(minDist2)
    print(overlap)
    print('final: ', minDist1 + minDist2 - 2* overlap)
 
def plotCities(data):
    """
    Plot city locations for TSP problem
    """
    X = [x[0] for x in data]
    Y = [x[1] for x in data]
    label = list(range(len(data)))
    fig, ax = plt.subplots()
    ax.scatter(X, Y)
    for i in label:
        ax.annotate(i, (X[i], Y[i]))
    plt.show()

def tsp(N, data):
    """
    Find the minimum path for Traveling Salesman Problem
    """
    ### Define function to convert row number to list of nodes
    def rowIdxToNodes(N, idx):
        string = ('{0:0' + str(N) + 'b}').format(idx)
        stringToList = list(string)
        stringToList[0] = '1' 
        nodes = [i for i,v in enumerate(stringToList) if v == '1']
        return nodes
    ### Define function to convert list of nodes to row number
    def nodesToRowIdx(N, nodes):
        nodesNoFirst = [x for x in nodes if x != 0]
        rowIdx = 0
        for node in nodesNoFirst:
            rowIdx += 2 ** (N - node - 1)
        return rowIdx
    ### Define function to calculate distance
    def getDist(a1, a2):
        dist = np.sqrt(((a1[0] - a2[0]) ** 2) + ((a1[1] - a2[1]) ** 2))
        return dist
    
    ### Init
    A = np.full((2 ** (N-1), N), np.inf)
    A[0, 0] = 0
    for m in range(2, N+1):
#    for m in range(3,4):
        possibleSs = []
        for i in range(2 ** (N-1)):
            nodes = rowIdxToNodes(N, i)
            if len(nodes) == m:
                possibleSs.append(nodes)
#        print(m)
#        print(possibleSs)
        for possibleS in possibleSs:
            for j in possibleS:
                if j != 0:
                    minVal = np.inf
                    rowIdxWithJ = nodesToRowIdx(N, possibleS)
                    rowIdxNoJ = nodesToRowIdx(N, [x for x in possibleS if x != j])
                    for k in possibleS:
                        if k != j:
                            distKJ = getDist(data[k], data[j])
                            minVal = min(minVal, A[rowIdxNoJ, k] + distKJ)
                    A[rowIdxWithJ, j] = minVal
    minDist = np.inf
    for j in range(1, N):
        minDist = min(minDist, A[2 ** (N-1) - 1, j] + getDist(data[j], data[0]))

    return minDist 

def week3():
    """
    Code up Traveling Salesman Problem using greedy algorithm for week 3
    """
    ### Initialization
    fileLocation = 'week-3/nn.txt'
#    fileLocation = 'week-3/nn-test.txt'

    ### Load data
    with open(fileLocation, 'r') as f:
        dataRaw = f.read().splitlines()
    numCity = int(dataRaw[0])
    dataV1 = [x.split() for x in dataRaw[1:]]
    dataV2 = [[float(y) for y in x] for x in dataV1]

#    print(dataV2)

    ### Apply greedy algorithm
    minDist = tspGreedy(numCity, dataV2)

    print(minDist)

def tspGreedy(N, data):
    """
    Greedy algorithm for tsp problem
    """
    def getDist(a1, a2):
        dist = np.sqrt(((a1[2] - a2[2]) ** 2) + ((a1[1] - a2[1]) ** 2))
        return dist
 
    ### Init
    totalDist = 0
    pointOrigin = data[0]
    pointAIdx = 1
    pointA = data[0]
    while len(data) != 1:
        localDistMin = np.inf
        for point in data:
            if point[0] != pointAIdx:
                tempDist = getDist(point, pointA)
                if tempDist < localDistMin:
                    localDistMin = tempDist
                    pointBIdx = point[0]
                    pointB = point
                elif tempDist == localDistMin:
                    if point[0] < pointBIdx:
                        pointBIdx = point[0]
                        pointB = point
        totalDist += localDistMin
        data = [x for x in data if x[0] != float(pointAIdx)]
        pointAIdx = pointBIdx
        pointA = pointB
#        print(totalDist)
#        print(pointAIdx)
#        print(pointA)
#        print(data)
    totalDist += getDist(pointOrigin, pointA)

    return totalDist      

if __name__ == '__main__':
#    week1()
#    week2()
    week3()























