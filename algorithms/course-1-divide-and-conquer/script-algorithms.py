### This covers all tasks from week 1.
### Charles Yinchu Dong
### 11/15/2018

### Import packages
import numpy as np
import re
import random

def countInversion(nums):
    ### Function that use merge sort to calculate inversion
    if len(nums) == 1:
        return nums, 0
    else:
        breakIdx = round(len(nums) / 2)
        part1, res1 = countInversion(nums[0:breakIdx])
        part2, res2 = countInversion(nums[breakIdx:])
        resMerge, resInversion = mergeCnt(part1, part2)
        invTotal = res1 + res2 + resInversion
        return resMerge, invTotal 

def mergeCnt(listA, listB):
    ### Function that merge two arrays. Return sorted version and count inversion.
    ### Initialization
    resList = []
    resCnt = 0

    ### Loop through two lists
    i , j, k = 0, 0, 0 
    lenA = len(listA)
    lenB = len(listB)
    while i < lenA and j < lenB:
        if listA[i] < listB[j]:
            resList.append(listA[i])
            i = i + 1
            # print('i = '+str(i))
        else:
            resList.append(listB[j])
            j = j + 1
            ### When move things from second list, count inversions.
            resCnt = resCnt + lenA - i
            # print('j = '+str(j))
    if i == lenA:
        resList = resList + listB[j:]
    else:
        resList = resList + listA[i:]

    return resList, resCnt

def week2Task():
    ### Initialization
    fileLocation = 'week-2/IntegerArray.txt'

    ### Load data
    with open(fileLocation, 'r') as f:
        dataRaw = f.read().splitlines()
    dataV1 = [int(x) for x in dataRaw]

#    print(dataV1)
    
    ### Call function to count inversions
    sortedList, numInversion = countInversion(dataV1)
    print('Sorted list: ', sortedList)
    print('Number of inversions: ', numInversion)

def week3Task():
    ### Initialization
    fileLocation = 'week-3/quickSort.txt'

    ### Load data
    with open(fileLocation, 'r') as f:
        dataRaw = f.read().splitlines()
    dataV1 = [int(x) for x in dataRaw]

#    print(dataV1)
    
    resList, resCnt = quickSort(dataV1, 'median')
    print('Number of comparisons: ', resCnt)

def partitionSubroutine(A, l, r, pivot):
    ### Function for partition subroutine.
    ### Initialization
    ### Swap pivot point to the most left
    if pivot == 'left':
        pivotIdx = l
    elif pivot == 'right':
        pivotIdx = r
    elif pivot == 'median':
        p1 = A[l]
        p2 = A[int(r/2)]
        p3 = A[r]
        pMedian = sorted([p1, p2, p3])[1]
        pivotIdx = A.index(pMedian)
    else:
        raise Exception('Invalid selection of pivot point')
    A[l], A[pivotIdx] = A[pivotIdx], A[l]

    ### Loop through list A
    p = A[l]
    i = l + 1
    for j in np.arange(l + 1, r + 1):
        if A[j] < p:
            A[i], A[j] = A[j], A[i]
            i = i + 1
    A[l], A[i - 1] = A[i - 1], A[l]

    return A[:i-1], [A[i-1]], A[i:], r

def quickSort(A, pivot):
    ### Function to do the quickSort
    ### Initialization
    if len(A) == 1 or len(A) == 0:
        return A, 0
    else:
        part1, partMiddle, part2, cntPartition = partitionSubroutine(A, 0, len(A)-1, pivot)
#        print(part1, partMiddle, part2, cntPartition)
        part1Sort, cntPart1 = quickSort(part1, pivot)
#        print(part1Sort, cntPart1)
        part2Sort, cntPart2 = quickSort(part2, pivot)
#        print(part2Sort, cntPart2)

        listAll = part1Sort + partMiddle + part2Sort
        cntAll = cntPartition + cntPart1 + cntPart2
        return listAll, cntAll 

def randomContract(adList):
    ### Function that uses randomiezed contraction algorithm
    ### Initializtion

    ### Pick a random edge
    numNodes = len(adList)
    node1Idx = random.randint(0, numNodes - 1)
    node1List = adList[node1Idx]
    numEdges = len(node1List) - 1
    edgeIdx = random.randint(1, numEdges)

    node1 = node1List[0]
    node2 = node1List[edgeIdx]

    node2List = [x for x in adList if x[0] == node2][0]

    ### Combine two nodes
    nodeCombineListRaw = node2List + node1List[1:]

    ### Eliminate self-loop
    nodeCombineListV1 = [x if x != node1 else node2 for x in nodeCombineListRaw]
    nodeCombineListV2 = [nodeCombineListV1[0]] + [x for x in nodeCombineListV1[1:] if x != node2] 

    ### Replace node1 in the remaining list
    remainListV1 = [x for x in adList if x != node1List and x != node2List]
    remainListV2 = [[x if x != node1 else node2 for x in y] for y in remainListV1]
    
    ### Check results
#    print(node1, node2)
#    print(nodeCombineListRaw)
#    print(nodeCombineListV1)
#    print(nodeCombineListV2)
#    print(remainListV1)
#    print(remainListV2)

    ### Return result
    res = remainListV2 + [nodeCombineListV2]
#    print(res)
    return res


def week4Task():
    ### Initialization
    fileLocation = 'week-4/countMinCut.txt'
    
    ### Load data
    with open(fileLocation, 'r') as f:
        dataRaw = f.read().splitlines()
    dataV1 = [re.split(r'\t+', x.rstrip('\t')) for x in dataRaw]
    dataV2 = [[int(x) for x in y] for y in dataV1] 

    #print(dataV2)
    
    ### Apply randomized contraction
    while len(dataV2) > 2:
        dataV2 = randomContract(dataV2)

    ### Count min cut
    minCut = len(dataV2[0]) - 1

    ### Return value
    print('Min cut: ', minCut)

if __name__ == '__main__':
    #week2Task()
    #print(mergeCnt([1,3, 6], [2,4,5]))
    #print(countInversion([1,3,6,2,4,5]))

#    week3Task()
#    print(partitionSubroutine([2,4,1,5,3,6,7,8,9], 0, 8, 'left'))
#    print(quickSort([2,4,1,5,3,6,7,8,9], 'left'))
#    print(quickSort([], 'left'))

    week4Task()
#    randomContract([[1,2,3], [2,1,3], [3,1,2,4], [4,3]])





































