### This covers all tasks from week 1.
### Charles Yinchu Dong
### 11/15/2018





#def countInversion(nums):
    ### Function that use merge sort to calculate inversion

def mergeCnt(listA, listB):
    ### Function that merge two arrays. Return sorted version and count inversion.
    ### Initialization
    resList = []
    resCnt = 0

    ### Loop through two lists
    i , j, k = 0, 0, 0 
    while i < len(listA):
        while j < len(listB):
            if listA[i] < listB[j]:
                resList.append(listA[i])
                i = i + 1
                print('i = '+str(i))
            else:
                resList.append(listB[j])
                j = j + 1
                print('j = '+str(j))
    if i == len(listA):
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
    countInversion(dataV1)



if __name__ == '__main__':
    #week2Task()
    mergeCnt([1,3, 6], [2,4,5])
