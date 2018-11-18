### This covers all tasks from week 1.
### Charles Yinchu Dong
### 11/15/2018





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
    fileLocation = 'week-1/IntegerArray.txt'

    ### Load data
    with open(fileLocation, 'r') as f:
        dataRaw = f.read().splitlines()
    dataV1 = [int(x) for x in dataRaw]

#    print(dataV1)
    
    ### Call function to count inversions
    sortedList, numInversion = countInversion(dataV1)
    print('Sorted list: ', sortedList)
    print('Number of inversions: ', numInversion)



if __name__ == '__main__':
    week2Task()
    #print(mergeCnt([1,3, 6], [2,4,5]))
    #print(countInversion([1,3,6,2,4,5]))
