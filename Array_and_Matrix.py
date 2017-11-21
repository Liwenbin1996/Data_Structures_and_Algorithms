import sys
import math
#转圈打印矩阵
def spiralOrderPrint(matrix):
    tR = tC = 0
    dR = len(matrix) - 1
    dC = len(matrix[0]) - 1
    while tR <= dR and tC <= dC:
        if tR == dR:
            for i in range(tC, dC+1):
                print(matrix[tR][i], end=' ')
        elif tC == dC:
            for i in range(tR, dR+1):
                print(matcix[i][tC], end=' ')
        else:
            for i in range(tC, dC):
                print(matrix[tR][i], end=' ')
            for i in range(tR, dR):
                print(matrix[i][dC], end=' ')
            for i in range(dC, tC, -1):
                print(matrix[dR][i], end=' ')
            for i in range(dR, tR, -1):
                print(matrix[i][tC], end=' ')
        tR += 1
        tC += 1
        dR -= 1
        dC -= 1


#将正方形矩阵顺指针转动90度
def rotateMatrix(m):
    tR = tC = 0
    dR = len(m) - 1
    dC = len(m[0]) - 1
    while tR <= dR:
        for i in range(tR, dR):
            m[tR][tC+i], m[tR+i][dC], m[dR][dC-i], m[dR-i][tC] \
                = m[dR-i][tC], m[tR][tC+i], m[tR+i][dC], m[dR][dC-i]
        tR = tC = tR + 1
        dR = dC = dR - 1


#之字型打印矩阵
def printMatrixZigZag(m):
    def printLevel(m, tR1, tC1, tR2, tC2, flag):
        if flag:
            while tR2 != tR1 - 1:
                print(m[tR2][tC2], end=' ')
                tR2 -= 1
                tC2 += 1
        else:
            while tR1 != tR2 + 1:
                print(m[tR1][tC1], end=' ')
                tR1 += 1
                tC1 -= 1


    tR1 = tC1 = tR2 = tC2 = 0
    dR = len(m) - 1
    dC = len(m[0]) - 1
    flag = True
    while tR1 <= dR and tC2 <= dC:
        printLevel(m, tR1, tC1, tR2, tC2, flag)
        tR1 = tR1 if tC1 != dC else tR1 + 1
        tC1 = tC1 + 1 if tC1 != dC else tC1
        tC2 = tC2 if tR2 != dR else tC2 + 1
        tR2 = tR2 + 1 if tR2 != dR else tR2
        flag = not flag
        

#在无序数组中找到最小的k个数
def getMinKNumsByHeap(arr, k):
    def heapInsert(heap, value, i):
        heap[i] = value
        parent = (i-1) // 2
        while parent >= 0:
            if arr[parent] < value:
                arr[i] = arr[parent]
                i = parent
                parent = (parent-1) // 2
            else:
                break
        arr[i] = value

    def heapify(heap, index):
        n = len(heap)
        child = 2 * index + 1
        tmp = heap[index]
        while child < n:
            if child < n-1 and heap[child] < heap[child+1]:
                child += 1
            if heap[child] > tmp:
                heap[index] = heap[child]
                index = child
                child = 2 * child + 1
            else:
                break
        heap[index] = tmp


    if arr == None or len(arr) == 0 or k < 1 or k > len(arr):
        return arr
    heap = [0 for i in range(k)]
    for i in range(k):
        heapInsert(heap, arr[i], i)
    for i in range(k, len(arr)):
        if arr[i] < heap[0]:
            heap[0] = arr[i]
            heapify(heap, 0)
    return heap


def getMinKNumsByBFPRT(arr, k):
    def getMinKthByBFPRT(arr, k):
        copyArr = [arr[i] for i in range(len(arr))]
        return select(copyArr, 0, len(copyArr)-1, k-1)

    def select(arr, begin, end, index):
        if begin == end:
            return arr[begin]
        pivot = medianOfMedian(arr, begin, end)
        pivotRange = partition(arr, begin, end, pivot)
        if index >= pivotRange[0] and index <= pivotRange[1]:
            return arr[index]
        elif index < pivotRange[0]:
            return select(arr, begin, pivotRange[0]-1, index)
        else:
            return select(arr, pivotRange[1]+1, end, index)

    def medianOfMedian(arr, begin, end):
        num = end - begin + 1
        offset = 0 if num % 5 == 0 else 1
        res = [0 for i in range(num//5 + offset)]
        for i in range(len(res)):
            start = begin + i * 5
            last = start + 4 if start + 4 <= end else end
            res[i] = median(arr, start, last)
        return select(res, 0, len(res) - 1, len(res) // 2)

    def median(arr, begin, end):
        insertSorts(arr, begin, end)
        mid = (begin + end) // 2
        return arr[mid] if (end - begin + 1) % 2 == 1 else arr[mid+1]

    def insertSorts(arr, begin, end):
        for i in range(begin+1, end+1):
            tmp = arr[i]
            j = i
            while j > begin and tmp < arr[j-1]:
                arr[j] = arr[j-1]
                j -= 1
            arr[j] = tmp

    def partition(arr, begin, end, pivot):
        small = begin - 1
        cur = begin 
        big = end + 1
        while cur != big:
            if arr[cur] == pivot:
                cur += 1
            elif arr[cur] < pivot:
                arr[small+1], arr[cur] = arr[cur], arr[small+1]
                small += 1
                cur += 1
            else:
                arr[cur], arr[big-1] = arr[big-1], arr[cur]
                big -= 1
        ran = []
        ran.append(small+1)
        ran.append(big-1)
        return ran



    if arr == None or len(arr) == 0 or k < 1 or k > len(arr):
        return arr
    minKth = getMinKthByBFPRT(arr, k)
    res = []
    for i in range(len(arr)):
        if arr[i] < minKth:
            res.append(arr[i])
    for i in range(len(res), k):
        res.append(minKth)
    return res


#需要排序的最短子数组长度
def getMinLength(arr):
    if arr == None or len(arr) < 2:
        return 0
    minOne = arr[-1]
    minIndex = -1
    for i in range(len(arr)-2, -1, -1):
        if arr[i] > minOne:
            minIndex = i
        else:
            minOne = arr[i]
    if minIndex == -1:
        return 0
    maxOne = arr[0]
    maxIndex = -1
    for i in range(1, len(arr)):
        if arr[i] < maxOne:
            maxIndex = i
        else:
            maxOne = arr[i]
    return maxIndex - minIndex + 1


#在数组中找到出现次数大于N/K的数
def printHalfMajor(arr):
    if arr == None or len(arr) == 0:
        print("No such number!")
        return
    cand = 0
    times = 0
    for i in range(len(arr)):
        if times == 0:
            cand = arr[i]
        elif arr[i] == cand:
            times += 1
        else:
            times -= 1
    times = 0
    for i in range(len(arr)):
        if arr[i] == cand:
            times += 1
    if times > len(arr) // 2:
        print(cand)
    else:
        print("No such number!")
        return


def printKMajor(arr, k):
    if arr == None or len(arr) == 0 or k < 2:
        print("No such number!")
        return
    map = {}
    moveMap = {}
    for i in range(len(arr)):
        if arr[i] in map:
            map[arr[i]] += 1
        else:
            if len(map) == k-1:
                for key in map:
                    map[key] -= 1
                    if map[key] == 0:
                        moveMap[key] = 1
                for key in moveMap:
                    del map[key]
                moveMap.clear()
            else:
                map[arr[i]] = 1
    times = 0
    flag = True
    for key in map:
        for i in range(len(arr)):
            if arr[i] == key:
                times += 1
        if times > len(arr) // k:
            print(key,end=' ')
            flag = False
        times = 0
    print("No such number!" if flag else '')


#在行列都排好序的矩阵中找数
def isContains(mat, k):
    if mat == None or len(mat) == 0 or len(mat[0]) == 0:
        return False
    row = 0
    col = len(mat[0]) - 1
    while row < len(mat) and col >= 0:
        if mat[row][col] == k:
            return True
        elif mat[row][col] < k:
            row += 1
        else:
            col -= 1
    return False


#最长的可整合子数组的长度
def getMaxIntegratedLength1(arr):
    def isIntergrated(arr):
        arr.sort()  #切片产生的是新列表，不需要克隆
        for i in range(1, len(arr)):
            if arr[i]-arr[i-1] != 1:
                return False
        return True

    if arr == None or len(arr) == 0:
        return 0
    length = 0
    for i in range(len(arr)):
        for j in range(i, len(arr)):
            if isIntergrated(arr[i : j+1]):
                length = max(length, j-i+1)
    return length


def getMaxIntegratedLength2(arr):
    if arr == None or len(arr) == 0:
        return 0
    length = 0
    map = {}
    for i in range(len(arr)):
        maxEle = -sys.maxsize
        minEle = sys.maxsize
        for j in range(i, len(arr)):
            if arr[j] in map:
                break
            maxEle = max(maxEle, arr[j])
            minEle = min(minEle, arr[j])
            if maxEle - minEle == j - i:
                length = max(length, j - i + 1)
        map.clear()
    return length



#不重复打印排序数组中相加和为给定值的二元组和三元组
def printUniquePair(arr, k):
    if arr == None or len(arr) < 2:
        return
    left = 0
    right = len(arr) - 1
    while left < right:
        sum = arr[left] + arr[right]
        if sum == k:
            if left == 0 or arr[left] != arr[left-1]:
                print(str(arr[left]) + "," + str(arr[right]))
                left += 1
                right -= 1
        elif sum > k:
            right -= 1
        else:
            left += 1


def printUniqueTriad(arr, k):
    if arr == None or len(arr) < 3:
        return
    for i in range(len(arr)):
        if i == 0 or arr[i] != arr[i-1]:
            left = i + 1
            right = len(arr) - 1
            while left < right:
                sum = arr[left] + arr[right]
                if sum == k - arr[i]:
                    if left == i+1 or arr[left] != arr[left-1]:
                        print(str(arr[i]) + "," + str(arr[left]) + "," + str(arr[right]))
                        left += 1
                        right -= 1
                elif sum > k - arr[i]:
                    right -= 1
                else:
                    left += 1


#未排序正数数组中累加和为指定值的最长子数组长度
def getMaxLength(arr, k):
    if arr == None or len(arr) == 0 or k < 1:
        return 0
    left = 0
    right = 0
    length = 0
    sum = arr[0]
    while left < len(arr) and right < len(arr):
        if sum == k:
            length = max(length, right-left+1)
            sum -= arr[left]
            left += 1
        elif sum > k:
            sum -= arr[left]
            left -= 1
        else:
            right += 1
            if right == len(arr):
                break
            sum += arr[right]
    return length


#未排序数组中累加和为给定值的最长子数组系列问题
def maxLength(arr, k):
    if arr == None or len(arr) == 0:
        return 0
    map = {}
    map[0] = -1
    sum = 0
    length = 0
    for i in range(len(arr)):
        sum += arr[i]
        if sum not in map:
            map[sum] = i
        if sum - k in map:
            length = max(length, i - map[sum-k])
    return length


#未排序数组中累加和小于或等于给定值的最长子数组问题
def maxLen(arr, k):
    def getLessIndex(h, num, index):
        left = 0
        right = index
        res = 1
        while left <= right:
            mid = (left + right) // 2
            if h[mid] >= num:
                res = mid
                right = mid - 1
            else:
                left = mid + 1
        return res

    if arr == None or len(arr) == 0:
        return 0
    sum = 0
    length = 0
    h = [0 for i in range(len(arr)+1)]
    for i in range(1, len(h)):
        sum += arr[i-1]
        h[i] = max(h[i-1], sum)
    sum = 0
    for i in range(len(arr)):
        sum += arr[i]
        pre = getLessIndex(h, sum-k, i)   #这个位置是h中的下表，注意转换成arr中的下标
        tmpLen = 0 if pre == -1 else i-pre+1
        length = max(length, tmpLen)
    return length


#计算数组的小和
def getSmallSum(arr):
    def mergeSort(arr, start, end):
        if start == end:
            return 0
        mid = (start + end) // 2
        return mergeSort(arr, start, mid) + mergeSort(arr, mid+1, end) + merge(arr, start, mid, end)

    def merge(arr, start, mid, end):
        left = start
        right = mid + 1
        res = []
        sum = 0
        while left <= mid and right <= end:
            if arr[left] < arr[right]:
                res.append(arr[left])
                sum += arr[left] * (end - right + 1)
                left += 1
            else:
                res.append(arr[right])
                right += 1
        res += arr[left : mid+1]
        res += arr[right : end+1]
        for i in range(start, end+1):
            arr[i] = res.pop(0)
        return sum


    if arr == None or len(arr) == 0:
        return 0
    return mergeSort(arr, 0, len(arr)-1)


#自然数组的排序
def sort1(arr):
    if arr == None or len(arr) == 0:
        return
    for i in range(len(arr)):
        while arr[i] != i+1:
            tmp = arr[arr[i]-1]
            arr[arr[i]-1] = arr[i]
            arr[i] = tmp
    return arr

def sort2(arr):
    if arr == None or len(arr) == 0:
        return
    for i in range(len(arr)):
        tmp = arr[i]
        while arr[i] != i+1:
            next = arr[tmp-1]
            arr[tmp-1] = tmp
            tmp = next
    return arr


#奇数下标都是奇数或者偶数下标都是偶数
def modify(arr):
    if arr == None or len(arr) < 2:
        return
    even = 0
    odd = 1
    while even < len(arr) and odd < len(arr):
        if arr[-1] & 1 == 0:
            arr[even], arr[-1] = arr[-1], arr[even]
            even += 2
        else:
            arr[odd], arr[-1] = arr[-1], arr[odd]
            odd += 2
    return arr


#子数组的最大累加和问题
def maxSum(arr):
    if arr == None or len(arr) == 0:
        return
    maxSum = -sys.maxsize
    curSum = 0
    for i in range(len(arr)):
        curSum += arr[i]
        maxSum = max(maxSum, curSum)
        curSum = curSum if curSum > 0 else 0
    return maxSum


#子矩阵的最大累加和问题
def matrixMaxSum(mat):
    if mat == None or len(mat) == 0 or len(mat[0]) == 0:
        return
    maxSum = -sys.maxsize
    for i in range(len(mat)):
        s = [0 for i in range(len(mat[0]))]
        for j in range(i, len(mat)):
            curSum = 0
            for k in range(len(mat[0])):
                s[k] += mat[j][k]
                curSum += s[k]
                maxSum = max(maxSum, curSum)
                curSum = curSum if curSum > 0 else 0
    return maxSum


#在数组中找到一个局部最小的位置
def indexOfLocalMin(arr):
    if arr == None or len(arr) == 0:
        return -1
    if len(arr) == 1 or arr[0] < arr[1]:
        return arr[0]
    if arr[-1] < arr[-2]:
        return arr[-1]
    left = 1
    right = len(arr) - 2
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] > arr[mid-1]:
            right = mid - 1
        elif arr[mid] > arr[mid+1]:
            left = mid + 1
        else:
            return mid


#数组中子数组的最大累乘积
def maxProduct(arr):
    if arr == None or len(arr) == 0:
        return 0
    maxPro = arr[0]
    minPro = arr[0]
    res = 0
    for i in range(1, len(arr)):
        maxPro = maxPro * arr[i]
        minPro = minPro * arr[i]
        maxPro = max(maxPro, minPro, arr[i])
        minPro = min(minPro, maxPro, arr[i])
        res = max(res, maxPro)
    return res


#打印N个数组整体最大的TopK
class HeapNode:
    def __init__(self, value, arrNum, index):
        self.value = value
        self.arrNum = arrNum
        self.index = index

def printTopK(matrix, k):
    def heapInsert(heap, i):
        if i == 0:
            return
        parent = (i-1) // 2
        tmp = heap[i]
        while tmp.value > heap[parent].value:
            heap[i] = heap[parent]
            i = parent
            parent = (i-1) // 2
        heap[i] = tmp

    def heapify(heap, i, n):
        child = 2 * i + 1
        tmp = heap[i]
        while child < n:
            if child < n-1 and heap[child].value < heap[child+1].value:
                child += 1
            if heap[child].value > tmp.value:
                heap[i] = heap[child]
                i = child
                child = 2 * i + 1
            else:
                break
        heap[i] = tmp


    if matrix == None or len(matrix) == 0:
        return []
    heapSize = len(matrix)
    heap = [0 for i in range(heapSize)]
    for i in range(len(heap)):
        index = len(matrix[i]) - 1
        heap[i] = HeapNode(matrix[i][index], i, index)
        heapInsert(heap,i)
    for i in range(k):
        if heapSize == 0:
            break
        print(heap[0].value, end=' ')
        if heap[0].index != 0:
            heap[0].value = matrix[heap[0].arrNum][heap[0].index-1]
            heap[0].index -= 1
        else:
            heap[0] = heap[-1]
            heapSize -= 1
        heapify(heap, 0, heapSize)


#边界都是１的最大正方形大小
def getMaxSize(mat):
    def setBorderMap(mat, right, down):
        row = len(mat) - 1
        col = len(mat[0]) - 1
        if mat[row][col] == 1:
            right[row][col] = 1
            down[row][col] = 1
        for i in range(len(mat)-2, -1, -1):
            if mat[i][col] == 1:
                right[i][col] = 1
                down[i][col] = down[i+1][col] + 1
        for j in range(len(mat[0])-2, -1, -1):
            if mat[row][j] == 1:
                right[row][j] = right[row][j+1] + 1
                down[row][j] = 1
        for i in range(len(mat)-2, -1, -1):
            for j in range(len(mat[0])-2, -1, -1):
                if mat[i][j] == 1:
                    right[i][j] = right[i][j+1] + 1
                    down[i][j] = down[i+1][j] + 1

    def hasSizeOfBorder(size, right, down):
        for i in range(len(mat) - size + 1):
            for j in range(len(mat[0]) - size + 1):
                if right[i][j] >= size and down[i][j] >= size and \
                        right[i+size-1][j] >= size and down[i][j+size-1] >= size:
                            return True
        return False


    if mat == None or len(mat) == 0 or len(mat[0]) == 0:
        return 0
    right = [[0 for i in range(len(mat[0]))] for j in range(len(mat))]
    down = [[0 for i in range(len(mat[0]))] for j in range(len(mat))]
    setBorderMap(mat, right, down)
    for size in range(min(len(mat), len(mat[0])), 0, -1):
        if hasSizeOfBorder(size, right, down):
            return size
    return 0


#不包含本位置值的累乘数组
def product1(arr):
    if arr == None or len(arr) < 2:
        return 
    allSum = 1
    count = 0
    for i in range(len(arr)):
        if arr[i] == 0:
            count += 1
        else:
            allSum *= arr[i]
    if count == 0:
        return [allSum // arr[i] for i in range(len(arr))]
    res = [0 for i in range(len(arr))]
    if count == 1:
        for i in range(len(arr)):
            if arr[i] == 0:
                res[i] = allSum
                break
    return res

def product2(arr):
    if arr == None or len(arr) < 2:
        return
    res = [0 for i in range(len(arr))]
    res[0] = arr[0]
    for i in range(1, len(res)):
        res[i] = res[i-1] * arr[i]
    tmp = 1
    for i in range(len(arr)-1, 0, -1):
        res[i] = res[i-1] * tmp
        tmp *= arr[i]
    res[0] = tmp
    return res


#数组的partiton调整
def leftUnique(arr):
    if arr == None or len(arr) < 2:
        return arr
    left = 0
    right = 1
    while right < len(arr):
        if arr[right] != arr[left]:
            arr[left+1], arr[right] = arr[right], arr[left+1]
            left += 1
            right += 1
        else:
            right += 1
    return arr


def sort(arr):
    if arr == None or len(arr) < 2:
        return arr
    left = -1
    right = len(arr)
    index = 0
    while index < right:
        if arr[index] == 0:
            arr[left+1], arr[index] = arr[index], arr[left+1]
            left += 1
        elif arr[index] == 1:
            index += 1
        else:
            arr[index], arr[right-1] = arr[right-1], arr[index]
            right -= 1
    return arr


#求最短通路值
def minPathValue(m):
    def walkTo(map, m, row, col, rQ, cQ, pre):
        if row < 0 or col < 0 or row == len(m) or col == len(m[0]) \
                or m[row][col] != 1 or map[row][col] != 0:
            return
        rQ.append(row)
        cQ.append(col)
        map[row][col] = pre + 1

    if m == None or len(m) == 0 or len(m[0]) == 0 \
            or m[0][0] != 1 or m[-1][-1] != 1:
        return 0
    map = [[0 for i in range(len(m[0]))] for j in range(len(m))]
    map[0][0] = 1
    rQ = []
    cQ = []
    rQ.append(0)
    cQ.append(0)
    while rQ:
        row = rQ.pop(0)
        col = cQ.pop(0)
        if row == len(m)-1 and col == len(m[0])-1:
            return map[-1][-1]
        walkTo(map, m, row+1, col, rQ, cQ, map[row][col])
        walkTo(map, m, row-1, col, rQ, cQ, map[row][col])
        walkTo(map, m, row, col+1, rQ, cQ, map[row][col])
        walkTo(map, m, row, col-1, rQ, cQ, map[row][col])
    return 0


#数组中未出现的最小正整数
def missNum(arr):
    if arr == None or len(arr) == 0:
        return
    left = 0
    right = len(arr)
    while left < right:
        if arr[left] == left + 1:
            left += 1
        elif arr[left] <= left or arr[left] > right or arr[arr[left]-1] == arr[left]:
            arr[left] = arr[right-1]
            right -= 1
        else:
            tmp = arr[left]
            arr[left] = arr[arr[left]-1]
            arr[tmp-1] = tmp
    return left + 1


#数组排序之后相邻数的最大差值
def maxGap(arr):
    def bucket(value, length, maxNum, minNum):
        return int((value - minNum) * length / (maxNum - minNum))

    if arr == None or len(arr) == 0:
        return 0
    length = len(arr)
    minNum = sys.maxsize
    maxNum = -sys.maxsize
    for i in range(len(arr)):
        minNum = min(minNum, arr[i])
        maxNum = max(maxNum, arr[i])
    if minNum == maxNum:
        return 0
    hasNum = [False for i in range(length+1)]
    maxs = [0 for i in range(length+1)]
    mins = [0 for i in range(length+1)]
    for i in range(len(arr)):
        index = bucket(arr[i], length, maxNum, minNum)
        maxs[index] = max(maxs[index], arr[i]) if hasNum[index] else arr[i]
        mins[index] = min(mins[index], arr[i]) if hasNum[index] else arr[i]
        hasNum[index] = True
    lastMax = maxs[0]
    res = 0
    for i in range(1, length+1):
        if hasNum[i]:
            res = max(res, mins[i] - lastMax)
            lastMax = maxs[i]
    return res

            




arr1 = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]
spiralOrderPrint(arr1)
print()
rotateMatrix(arr1)
print(arr1)
printMatrixZigZag(arr1)
print()
print(getMinKNumsByHeap([9,8,7,6,5,4,3,2,1], 3))
print(getMinKNumsByBFPRT([9,7,6,4,3,2,1,5], 3))
print(getMinLength([1,5,3,4,2,6,7]))
printHalfMajor([1,2,3,4,5,1,1,1,1])
printKMajor([1,2,1,3,4,5,1,1,1,1,1,1,1,1,1], 3)
print(isContains([[1,2,3], [4,5,6], [7,8,9]], 0))
print(getMaxIntegratedLength1([5,5,3,2,6,4,3]))
print(getMaxIntegratedLength2([5,5,3,2,6,4,3]))
printUniquePair([-8, -4, -3, 0, 1, 2, 4, 5, 8, 9], 10)
printUniqueTriad([-8, -4, -3, 0, 1, 2, 4, 5, 8, 9], 10)
print(getMaxLength([1,2,1,1,1], 3))
print(maxLength([1,1,-2,1,3,4,5], 4))
print(maxLen([3, -2, -4, 0, 6], -2))
print(getSmallSum([1,3,5,2,4,6]))
print(sort1([1,2,5,3,4]))
print(sort2([1,2,5,3,4]))
print(modify([1,2,3,4,5,6]))
print(maxSum([1,-2,3,5,-2,6,-1]))
print(matrixMaxSum([[-90,48,78], [64,-40,64], [-81,-7,66]]))
print(indexOfLocalMin([2,3,1,4,5,6]))
print(maxProduct([-2.5, 4, 0, 3, 0.5, 8, -1]))
printTopK([[219,405,538,845,971],[148,558],[52,99,348,691]], 5)
print()
print(getMaxSize([[0,1,1,1,1], [0,1,0,0,1], [0,1,0,0,1], [0,1,1,1,1], [0,1,0,1,1]]))
print(product1([2,3,1,4]))
print(product2([2,3,1,4]))
print(leftUnique([1,2,2,2,3,3,4,5,6,6,7,7,8,8,8,9]))
print(sort([1,0,2,0,2,0,1,2,0,2,1,1,2]))
print(minPathValue([[1,0,1,1,1], [1,0,1,0,1], [1,1,1,0,1], [0,0,0,0,1]]))
print(missNum([4,3,2,1]))
print(maxGap([9,3,1,10]))
