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

if __name__ == '__main__':
    week1()























