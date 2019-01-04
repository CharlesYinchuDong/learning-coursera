### This covers all tasks for course 3
### Charles Yinchu Dong
### 01/01/2019

### Import packages
import numpy as np
import re
import random
import copy
from collections import Counter
import sys, threading
import heapq

def week1Task1():
    """
    Code up scheduling problem
    """
    ### Initialization
    fileLocation = 'week-1/jobs.txt'

    res = []

    ### Load data
    with open(fileLocation, 'r') as f:
        dataRaw = f.read().splitlines()
    dataV1 = [x.split() for x in dataRaw[1:]]
    dataV2 = [[int(y) for y in x] for x in dataV1]
    numTotal = int(dataRaw[0])
    
#    print(dataRaw)
#    print(dataV2)
#    print(numTotal)
    
    jobSchedule(total=numTotal, jobs=copy.deepcopy(dataV2), method='-')
#    jobSchedule(total=numTotal, jobs=copy.deepcopy(dataV2), method='/')
    
def jobSchedule(total, jobs, method):
    """
    Function that schedule the job given certain cost function
    """
    ### Initialization
    res = 0

    ### Calculate compare value based on methods
    if method == '-':
        [x.append(x[0] - x[1]) for x in jobs]
    elif method == '/':
        [x.append(x[0] / x[1]) for x in jobs]
    else:
        raise Exception('Unknown method')

    ### Sort the job list
    jobsV1 = sorted(jobs, key=lambda x: (x[2], x[0]), reverse=True)
    
    ### Calculate completion times
    subTime = [x[1] for x in jobsV1]
    completeTime = list(np.cumsum(subTime))
    
    ### Get weights
    weights = [x[0] for x in jobsV1]

    ### Calculate weighted sum
    for i in range(total):
        res = res + weights[i] * completeTime[i]

    print(jobs)
    print(jobsV1)
    print(completeTime)
    print(res)

def week1Task2():
    """
    Code up MST problem.
    """
    ### Initialization
    fileLocation = 'week-1/edges.txt'
#    fileLocation = 'week-1/edges-test1.txt'
#    fileLocation = 'week-1/edges-test2.txt'

    ### Load data
    with open(fileLocation, 'r') as f:
        dataRaw = f.read().splitlines()
    dataV1 = [x.split() for x in dataRaw[1:]]
    dataV2 = [[int(y) for y in x] for x in dataV1]
    N = dataRaw[0].split()[0]
    M = dataRaw[0].split()[1]

#    print(dataRaw)
#    print(dataV2)
#    print(N)
#    print(M)
 
    findMST(n=N, m=M, edges=copy.deepcopy(dataV2))

def findMST(n, m, edges):
    """
    Function that calculate minimum spanning tree and calculate total costs
    """
    ### Initialization
    known = set([edges[0][0]])
    knownEdge = []

    maxEdge = max([x[2] for x in edges]) * 10
    
    allNodes = set([x[0] for x in edges] + [x[1] for x in edges])

    vMap = dict.fromkeys(allNodes, maxEdge)

    ### Heapify
    for edge in edges:
        if edge[0] == list(known)[0]:
            vMap[edge[1]] = edge[2]
        elif edge[1] == list(known)[0]:
            vMap[edge[0]] = edge[2]

    vMap.pop(list(known)[0])
    edgeValues = list(vMap.values())    
    heapq.heapify(edgeValues)

    ### While loop until spanning all nodes
    while known != allNodes:
        ### Pop up from heap
        minEdge = heapq.heappop(edgeValues)

        ### Retrieve corresponding node
        minNode = [x for x in vMap if vMap[x] == minEdge][0]

        ### Update known with node and edge
        known.add(minNode)
        knownEdge.append(minEdge)
        vMap.pop(minNode)

        ### Update the heap
        associate = {}
        for edge in edges:
            if edge[0] == minNode:
                associate[edge[1]] = edge[2]
            elif edge[1] == minNode:
                associate[edge[0]] = edge[2]

#        print('known: ', known)
#        print('knownEdge: ', knownEdge)
#        print('vMap1: ', vMap)
#        print(minNode)
#        print(minEdge)
#        print('associate: ', associate)

        for w in associate:
            if w not in known:
                ### Delete from heap
                edgeValues[edgeValues.index(vMap[w])] = edgeValues[-1]
                edgeValues.pop()
                heapq.heapify(edgeValues)
    
                ### Compute relative minimum
                relaMin = min(vMap[w], associate[w])
    
                ### Update vertex-value map
                vMap[w] = relaMin
    
                ### Insert back to heap
                heapq.heappush(edgeValues, relaMin)
        
#        print('edgeValues', edgeValues)
#        print('vMap2: ', vMap)

#    print(vMap)
#    print(edgeValues)
    print(sum(knownEdge))
    

if __name__ == '__main__':
#    week1Task1()
    week1Task2()


















