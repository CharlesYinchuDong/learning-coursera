### This covers all tasks for course 2
### Charles Yinchu Dong
### 12/01/2018

### Import packages
import numpy as np
import re
import random
import copy
from collections import Counter
import sys, threading
import heapq

### Chagne recursion depth
sys.setrecursionlimit(800000)
threading.stack_size(67108864)

def week1Task():
    ### Initialization
    fileLocation = 'week-1/scc.txt'
#    fileLocation = 'week-1/testCase1.txt'

    ### Load data
    with open(fileLocation, 'r') as f:
        dataRaw = f.read().splitlines()
    dataV1 = [x.split() for x in dataRaw]
    dataV2 = [[int(x) for x in y] for y in dataV1]

    ### Get all nodes
    allNodes = set([x for y in dataV2 for x in y])

    ### Convert list to dict
    dataV3 = {x:set() for x in allNodes}
    for (key, value) in dataV2:
        if key in dataV3:
            dataV3[key].add(value)
        else:
            dataV3[key] = set([value])

    ### Calculate reverse graph
    dataCopy = copy.deepcopy(dataV2)
    dataV4 = list(map(graphReverse, dataCopy))

    ### Convert list to dict
    dataV5 = {x:set() for x in allNodes}
    for (key, value) in dataV4:
        if key in dataV5:
            dataV5[key].add(value)
        else:
            dataV5[key] = set([value])

    ### Apply Kosarajus Two Pass algorithm
    sccs = kosarajusTwoPass(dataV3, dataV5)

#    print(dataRaw)
#    print(dataV3)
#    print(dataV5)
    print(sccs)
    
def graphReverse(a):
    a[0], a[1] = a[1], a[0]
    return a

def kosarajusTwoPass(G, GRev):
    ### Function that apply Kosarajus Two Pass algorithm
    ### Initialization

    ### Apply DFS-Loop on G reverse
    seq1 = sorted(GRev.keys(), reverse=True)
    leader1, f1 = dfsLoop(GRev, seq = seq1)

    ### Apply DFS-loop on G
    seq2 = [x[0] for x in sorted(f1.items(), key = lambda pair: pair[1], reverse=True)]
    leader2, f2 = dfsLoop(G, seq = seq2)

    ### Calculate scc
    leaderSort = sorted(leader2.items(), key=lambda pair: len(pair[1]), reverse=True)
    topLeader = [len(x[1]) for x in leaderSort[0:5]]

    ### Check the results
#    print(G)
#    print(GRev)
#    print(leader1, f1)
#    print(leader2, f2)
#    print(sccsTop)

    ### Return
    return topLeader

def dfsLoop(G, seq):
    ### Function for DFS-Loop
    ### Initialization
    global t, s, explored, leader, fValues
    t = 0
    s = 0

    explored = set()
    leader = dict() 
    fValues = dict() 

    ### Loop through all nodes following sequence order
    for i in seq:
        if i not in explored:
            s = i
            dfs(G, i)

    ### Check results
#    print('Seq: ', seq)
#    print('G:', G)
#    print('explored: ', explored)
#    print('leader: ', leader)
#    print('fValues: ', fValues)

    ### Return results
    return leader.copy(), fValues.copy()

def dfs(G, i):
    ### Function for dfs
    ### Initialization
    global t, s, explored, leader

    ### Mark i as explored
    explored.add(i)

    ### Set leader
    if s in leader:
        leader[s].add(i)
    else:
        leader[s] = set([i])

    ### Explore all edges
    js = G[i]
    for j in js:
        if j not in explored:
            dfs(G, j)

    ### Assign t value
    t = t + 1
    fValues[i] = t

def test1():
    global a, b, c
    a = 1
    print(a)
    test2()
    print(a)

def test2():
    global a
    a = 2
    print(a)


def week2Task():
    """
    Code up dijkstras shorted path algorithm
    """
    ### Initialization
    fileLocation = 'week-2/shorted-path.txt'
#    fileLocation = 'week-2/testCase2.txt'

    s = 1
    t = [7,37,59,82,99,115,133,165,188,197]

    res = []

    ### Load data
    with open(fileLocation, 'r') as f:
        dataRaw = f.read().splitlines()
    dataV1 = [re.split(r'\t+', x.rstrip('\t')) for x in dataRaw]
    dataV2 = {int(x[0]): x[1:] for x in dataV1}
    dataV3 = dict()
    for key in dataV2.keys():
        valueV1 = [x.split(',') for x in dataV2[key]]
        valueV2 = {int(x[0]): int(x[1]) for x in valueV1}
        dataV3[key] = valueV2

#    print(dataRaw)
#    print(dataV1)
#    print(dataV2)
#    print(dataV3)

    ### Pass to dijkstras function
    for tSub in t:
        resSub = dijkstras(copy.deepcopy(dataV3), s=1, t=tSub)
        res.append(resSub) 

    print(res)

def dijkstras(data, s, t):
    """
    Perform dijkstra's shorted path algorithm
    """
#    print(data,s,t)

    ### Initialization
    nodesAll = set(data.keys())
    for i in data.items():
        nodesAll.update(list(i[1].keys()))

    ### Update data with nodes that no outcome
    for node in nodesAll:
        if node not in data:
            data[node] = dict()

    ### Initialization
    X = set([s])
    A = {s: 0}
    B = [s]

    ### Calculate all - X
    Y = nodesAll.difference(X)

    ### Constractu heap
    YInfo = data[s]
    for v in Y:
        if v not in YInfo:
            YInfo[v] = 1000000
    heap = [x[1] for x in YInfo.items()]
    heapq.heapify(heap)

    while t not in X:
        ### Pop and add to X
        wValue = heapq.heappop(heap)

        w = [x[0] for x in YInfo.items() if x[1] == wValue][0]

        X.add(w)
        A[w] = wValue
        YInfo.pop(w)
        ### Take out w-related edges and put it back
        for v in data[w]:
            if v not in X:
                heap.pop(heap.index(YInfo[v]))
                vValue1 = YInfo[v]
                vValue2 = A[w] + data[w][v]
                vValue = min(vValue1, vValue2)
                YInfo[v] = vValue
                heapq.heappush(heap, vValue)
                heapq.heapify(heap)

    return A[t]

def week3Task():
    """
    Code up median maintanence
    """
    ### Initialization
    fileLocation = 'week-3/median.txt'

    res = []

    ### Load data
    with open(fileLocation, 'r') as f:
        dataRaw = f.read().splitlines()
    dataV1 = [int(x) for x in dataRaw]

#    dataV1 = [1,3,4,2,5,11,6,8,12]
 
#    print(dataRaw)
#    print(dataV1)

    ### Initialize two heap
    heapH = []
    heapL = []
    heapq.heapify(heapH)
    heapq.heapify(heapL)

    ### Loop through all elements
    for i in range(len(dataV1)):
        v = dataV1[i]
        ### If it is first one, add to low heap
        if i == 0:
            heapq.heappush(heapL, -1 * v)
            m = v
        else:
            ### Compare v with the largest value in low heap
            if v > -1 * heapq.nsmallest(1, heapL)[0]:
                heapq.heappush(heapH, v)
            else:
                heapq.heappush(heapL, -1 * v)

            ### Balance low heap and high heap
            if len(heapL) > len(heapH) + 1:
                adjust = -1 * heapq.heappop(heapL)
                heapq.heappush(heapH, adjust)
            elif len(heapH) > len(heapL) + 2:
                adjust = heapq.heappop(heapH)
                heapq.heappush(heapL, -1 * adjust)

            ### Based on size of both heap, select median
            if len(heapL) >= len(heapH):
                m = -1 * heapq.nsmallest(1, heapL)[0]
            else:
                m = heapq.nsmallest(1, heapH)[0]

        res.append(m)

#    print(res)
    print('Result: ', sum(res) % 10000)

def week4Task():
    """
    Task for week 4
    """
    ### Initialization
    fileLocation = 'week-4/2sum.txt'

    LOW = -10000
    HIGH = 10000

    res = 0

    ### Load data
    with open(fileLocation, 'r') as f:
        dataRaw = f.read().splitlines()
    dataV1 = [int(x) for x in dataRaw]
    dataV2 = set(dataV1)

#    print(dataV1)

    ### Sort data first
    dataSort = sorted(dataV2)

    ### Loop through all elements to find possible sums
    tAll = []
    for i in dataSort:
        boundLow = LOW - i
        boundHigh = HIGH - i
        idxLow = binarySearchLow(dataSort, boundLow)
        idxHigh = binarySearchHigh(dataSort, boundHigh) -1 
        jList = dataSort[idxLow : idxHigh + 1]
        tList = [x + i for x in jList]
        tAll.append(tList)

    ### Count distinct t values
    tFlat = [x for y in tAll for x in y]
    tSet = set(tFlat)

    ### Distinct t values
    tNum = len(tSet)

    print(tNum)

def binarySearch(arr, x, l, r):
    """
    Implement binary search algorithm
    """
    ### Initialization

    ### To see if x exit
    if r >= l: 
        ### Compare x with middle value
        mid = l + int((r - l) / 2)
        if arr[mid] == x:
            return mid
        elif arr[mid] > x:
            return binarySearch(arr, x, l, mid-1)
        else:
            return binarySearch(arr, x, mid+1, r)
    else:
        return -1

def binarySearchLow(arr, x, lo=0, hi=None):
    """
    Return the index of the value larger or equal than x
    """
    if hi == None:
        hi = len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] == x:
            return mid
        elif x > arr[mid]:
            lo = mid + 1
        else:
            hi = mid
    return lo

def binarySearchHigh(arr, x, lo=0, hi=None):
    """
    Return the index of the value larger or equal than x
    """
    if hi == None:
        hi = len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < arr[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo

if __name__ == '__main__':
#    thread = threading.Thread(target=week1Task)
#    thread.start()
#    testCase = [[1,5],[2,3],[3,4],[4,2],[4,5],[5,6],[6,9],[6,1],[7,8],[8,9],[9,7]]
#    testCase = [[1,4],[2,8],[3,6],[4,7],[5,2],[6,9],[7,1],[8,6],[8,5],[9,7],[9,3],[10,1]]
#    kosarajusTwoPass(testCase)
#    test1()
#    testCase = [[1,7],[2,5],[3,9],[4,1],[5,8],[6,3],[6,8],[7,9],[7,4],[8,2],[9,6]]
#    dfsLoop(testCase, list(reversed(np.arange(1, 10))))

#    week2Task()

#    week3Task()

    week4Task()
#    arr = [2,4,10,13]
#    print(binarySearch(arr, 6, 0, len(arr)-1))
#    print(binarySearchLow(arr, 4))
#    print(bisect_left(arr, 0))
#    print(binarySearchHigh(arr,3))





















