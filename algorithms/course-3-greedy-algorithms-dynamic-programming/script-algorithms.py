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
#    dataV1 = [int(x) for x in dataRaw]
    
    print(dataRaw)

if __name__ == '__main__':
    week1Task1()


















