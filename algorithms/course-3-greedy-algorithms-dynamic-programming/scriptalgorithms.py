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
    

def week2Task1():
    """
    Code up first clustering problem with max-spacing.
    """
    ### Initialization
    fileLocation = 'week-2/clustering1.txt'
#    fileLocation = 'week-2/p1_test.txt'
    
    K = 4

    ### Load data
    with open(fileLocation, 'r') as f:
        dataRaw = f.read().splitlines()
    dataV1 = [x.split() for x in dataRaw[1:]]
    dataV2 = [[int(y) for y in x] for x in dataV1]
    N = int(dataRaw[0])

    ### Initialize union-find structure
    uf = UF(N)

    ### Sort by edge weights
    dataV3 = sorted(dataV2, key=lambda x: x[2], reverse=False)

#    print(dataV3)

    k = N
    i = 0
    nodes = list(range(N))
    while k >= K:
        ### Union the two nodes with least edge
        edgeInfo = dataV3[i]
        if uf.find(edgeInfo[0]-1) != uf.find(edgeInfo[1]-1):
            uf.union(edgeInfo[0]-1, edgeInfo[1]-1)

        ### Check the remaining clastering
        zipped = list(zip(nodes, uf._id))
        k = len(set([y for (x, y) in zipped if x == y]))
#        print(k)
#        print(uf._id)
#        print(uf._rank)
#        print(i)
#        print('\n\n')

        ### Increase
        i += 1


#    print(dataRaw)
#    print(dataV2)
#    print(dataV3)
    print(dataV3[i-1])
    
def week2Task2():
    """
    Code up first clustering problem with max-spacing.
    """
    ### Initialization
    fileLocation = 'week-2/clustering_big.txt'
    
    ### Load data
    with open(fileLocation, 'r') as f:
        dataRaw = f.read().splitlines()
    dataV1 = [x.replace(" ", "") for x in dataRaw[1:]]
    dataSet = set(dataV1)
    dataV2 = list(set(dataV1))
    N = len(dataV2)
    BITS = int(dataRaw[0].split()[1])

    ### Build dict for nodes
    nodeDict = {}
    for i in range(N):
        nodeDict[dataV2[i]] = i

    ### Initialize Union-Find
    uf = UF(N)

    ### Loop through all nodes
    for i in range(N):
        ### Get nodes with distance 1 or 2
        dist1 = findNode(dataV2[i], 1)
        dist2 = findNode(dataV2[i], 2)
        dist12Raw = dist1 + dist2
        dist12 = []
        for j in dist12Raw:
            if j in dataSet:
                dist12.append(j)

        ### Union them together
        iIndex = nodeDict[dataV2[i]]
        for j in dist12:
            jIndex = nodeDict[j]
            if uf.find(iIndex) != uf.find(jIndex):
                uf.union(iIndex, jIndex)

    nodesAll = list(range(N))
    numCluster = len([y for (x,y) in zip(nodesAll, uf._id) if x == y])


    ### Testing
    print(N)
    print(BITS)
#    print(dataV2)
#    print(len(dataV2))
    print(numCluster)

def findNode(node, dist=1):
    """ Find nodes with certain Hamming distance """
    ### Initialization
    BITS = len(node)

    res = []

    ### Calculate possible nodes
    if dist == 1:
        for i in range(BITS):
            temp = list(node)
            temp[i] = '1' if temp[i] == '0' else '0'
            resSub = "".join(temp)
            res.append(resSub)
    elif dist == 2:
        for i in range(BITS):
            for j in range(BITS):
                temp = list(node)
                if i != j:
                    temp[i] = '1' if temp[i] == '0' else '0'
                    temp[j] = '1' if temp[j] == '0' else '0'
                    resSub = "".join(temp)
                    res.append(resSub)
        res = list(set(res))
    else:
        raise Exception('We donnot support this distance!')

    return res

class UF:
    """ Class for union find data structure """

    def __init__(self, N):
        """ Initialization """
        self._N = N
        self._id = list(range(N))
        self._rank = [0] * N

    def find(self, x):
        """ Find the leader of x """
        idd = self._id
        leader = idd[x]

        while idd[leader] != leader:
            leader = idd[leader]

        ### Path compression
        idd[x] = leader

        return leader

    def union(self, p, q):
        """ Union p and q """
        idd = self._id
        rank = self._rank

        ### Find leaders
        i = self.find(p)
        j = self.find(q)

        ### Union by rank
        if rank[i] == rank[j]:
            idd[j] = idd[i]
            rank[i] = rank[i] + 1
        elif rank[i] > rank[j]:
            idd[j] = idd[i]
        else:
            idd[i] = idd[j]

def week3Task1():
    """ Code up huffman code """
    ### Initialization

    sys.setrecursionlimit(1500)

    fileLocation = 'week-3/huffman.txt'
#    fileLocation = 'week-3/huffman_test1.txt'
    
    ### Load data
    with open(fileLocation, 'r') as f:
        dataRaw = f.read().splitlines()
    dataV1 = [int(x) for x in dataRaw]
    cntSymbol = dataV1[0]
    A = dataV1[1:]

#    print(dataV1)

    ### Recursivly calculate haffman code
    res = [''] * cntSymbol
    f = {}
    for i in range(cntSymbol):
        if A[i] in f:
            f[A[i]].append(i)
        else:
            f[A[i]] = [i]
    heapq.heapify(A)
    HUF(A, f, res)

    resFinal = res[0:cntSymbol]

    minLen = min([len(x) for x in resFinal])
    maxLen = max([len(x) for x in resFinal])

    print(res)
    print(A)
    print(f)
    print(resFinal)
    print('Min length: ', minLen)
    print('Max length: ', maxLen)

def HUF(A, f, res):
    """ Haffman function """
    ### Initialization
    ### Recursion
    if len(A) == 2:
        res[list(f.values())[0][0]] = '0'
        res[list(f.values())[1][0]] = '1'
    else:
        ### Pop two min weights
        u = heapq.heappop(A)
        v = heapq.heappop(A)

        ### Delete two vertices in map f
        if len(f[u]) == 1:
            uIdx = f[u][0]
            del f[u]
        else:
            uIdx = f[u].pop()
        if len(f[v]) == 1:
            vIdx = f[v][0]
            del f[v]
        else:
            vIdx = f[v].pop()
    
        ### Add up to new weights
        z = u + v
        zIdx = len(res)
        res.append('')

#        print(u, v, uIdx, vIdx, z, zIdx)

        ### Insert back new value
        heapq.heappush(A, z)
        if z in f:
            f[z].append(zIdx)
        else:
            f[z] = [zIdx]

        ### Recursive on HUF
        HUF(A, f, res)

        ### Reverse the tree
        res[uIdx] = res[zIdx] + '0'
        res[vIdx] = res[zIdx] + '1'      

def week3Task2():
    """ Code up huffman code """
    ### Initialization
    sys.setrecursionlimit(1500)

    fileLocation = 'week-3/mwis.txt'
    
    ### Load data
    with open(fileLocation, 'r') as f:
        dataRaw = f.read().splitlines()
    dataV1 = [int(x) for x in dataRaw]
    cntV = dataV1[0]
    dataV2 = dataV1[1:]

#    print(dataV2)

    ### Init for dynamic programming
    A = [0] * (cntV+1)
    A[0] = 0
    A[1] = dataV2[0]
    S = []

    ### Get Maximum-weight value
    for i in range(2, cntV+1):
        A[i] = max(A[i-1], A[i-2] + dataV2[i-1])

    ### Get the solution
    j = cntV
    while j >= 1:
        if A[j-1] >= A[j-2] + dataV2[j-1]:
            j -= 1
        else:
            S.append(j)
            j -= 2

    ### Check for specific vertices
    vCheck = [1, 2, 3, 4, 17, 117, 517, 997]
    resCheck = [1 if x in S else 0 for x in vCheck]

    print(resCheck)
    print(''.join([str(e) for e in resCheck]))

if __name__ == '__main__':
#    week1Task1()
#    week1Task2()
#    week2Task1()
#    week2Task2()
#    print(findNode('11111', 2))

#    week3Task1()
    week3Task2()















