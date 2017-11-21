import math

def insertSort(arr):
    if len(arr) <= 1:
        return arr
    for i in range(1,len(arr)):
        tmp = arr[i]        
        j = i 
        while j > 0 and tmp < arr[j-1]:
            arr[j] = arr[j-1]
            j -= 1
        arr[j] = tmp
    print(arr) 

def shellSort(arr):
    if len(arr) <= 1: 
        return arr
    Increment = len(arr) // 2
    while Increment > 0:
        for i in range(Increment, len(arr)):
            tmp = arr[i]
            j = i
            while j > 0 and tmp < arr[j-Increment]:
                arr[j] = arr[j-Increment]
                j -= Increment
            arr[j] = tmp
        Increment = Increment // 2
    print(arr)

def heapSort(arr):
    def precDown(a, i, N):
        child = 2 * i + 1
        tmp = a[i]
        while child < N:
            if child < N-1 and a[child] < a[child+1]:
                child += 1
            if tmp < a[child]:
                a[i] = a[child]
                i = child
            else:
                break
            child = child * 2 + 1
        a[i] = tmp

    if len(arr) <= 1:
        return arr
    N = len(arr)
    for i in range((N-2)//2, -1, -1):
        precDown(arr,i,N)
    for i in range(N-1, 0, -1):
        arr[0], arr[i] = arr[i],arr[0]
        precDown(arr,0,i)
    print(arr)

def mergeSort(arr):
    def Msort(a, tmplist, left, right):
        if left < right:
            center = (left + right) // 2
            Msort(a, tmplist, left, center)
            Msort(a, tmplist, center+1, right)
            Merge(a, tmplist, left, center+1, right)
    def Merge(a, tmplist, Lpos, Rpos, Rend):
        Lend = Rpos - 1
        TmpPos = Lpos
        NumElements = Rend - Lpos + 1
        while Lpos <= Lend and Rpos <= Rend:
            if a[Lpos] < a[Rpos]:
                tmplist[TmpPos] = a[Lpos]
                TmpPos += 1
                Lpos += 1
            else:
                tmplist[TmpPos] = a[Rpos]
                TmpPos += 1
                Rpos += 1
        while Lpos <= Lend:
            tmplist[TmpPos] = a[Lpos]
            Lpos += 1
            TmpPos += 1
        while Rpos <= Rend:
            tmplist[TmpPos] = a[Rpos]
            Rpos += 1
            TmpPos += 1
        for i in range(NumElements):
            a[Rend] = tmplist[Rend]
            Rend -= 1

    if len(arr) <= 1:
        return arr
    tmparray = [0]*len(arr)
    Msort(arr, tmparray, 0, len(arr)-1)
    print(arr)

def MergeSort(arr):
    def Merge(left, right):
        i, j = 0, 0
        result = []
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result += left[i:]
        result += right[j:]
        return result

    if len(arr) <= 1:
        return arr
    center = len(arr) // 2
    left = MergeSort(arr[:center])
    right = MergeSort(arr[center:])
    return Merge(left, right)

def quickSort(arr):
    def Qsort(a, left, right):
        i = left
        j = right
        if i >= j:
            return
        key = Median(a, left, right)
        while i < j:
            while i < j and a[j] >= key:
                j -= 1
            a[i] = a[j]
            while i < j and a[i] <= key:
                i += 1
            a[j] = a[i]
        a[i] = key
        Qsort(a, left, i-1)
        Qsort(a, i+1, right)

    def Median(a, left, right):
        center = (left + right) // 2
        if a[left] > a[center]:
            a[left], a[center] = a[center], a[left]
        if a[center] > a[right]:
            a[center], a[right] = a[right], a[center]
        if a[left] > a[right]:
            a[left], a[right] = a[right], a[left]
        a[center], a[left] = a[left], a[center]
        return a[left]

    if len(arr) <= 1:
        return arr
    Qsort(arr, 0, len(arr)-1)
    print(arr)

def bubbleSort(arr):
    length = len(arr)
    if length <= 1:
        return arr
    while length > 1:
        for i in range(length-1):
            if arr[i] > arr[i+1]:
                arr[i], arr[i+1] = arr[i+1], arr[i]
        length -= 1
    print(arr)

def selectSort(arr):
    if len(arr) <= 1:
        return arr
    for i in range(len(arr)-1):
        min = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min]:
                min = j
        arr[i], arr[min] = arr[min], arr[i]
    print(arr)

def bucketSort(arr, radix=10):
    if len(arr) <= 1:
        return arr
    K = int(math.ceil(math.log(max(arr), radix)))
    bucket = [[] for i in range(radix)]
    for i in range(1, K+1):
        for element in arr:
            bucket[element%(radix**i)//(radix**(i-1))].append(element)
        del(arr[:])
        for each in bucket:
            arr.extend(each)
        bucket = [[] for i in range(radix)]
    print(arr)

test1 = []
test2 = [1]
test3 = [9,7,5,4,10,31,3,2,1,13,14]
heapSort(test3)
