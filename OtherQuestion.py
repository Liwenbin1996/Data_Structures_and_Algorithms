import sys 
import copy
import random 
import math 
#从５随机到７随机及其扩展
def rand1To5():
    return int(random.random()*5) + 1

def rand1To7():
    num = (rand1To5()-1) * 5 + rand1To5() - 1
    while num > 20:
        num = (rand1To5()-1) * 5 + rand1To5() - 1
    return num % 7 + 1


def rand01p():
    p = 0.83
    return 0 if random.random() < p else 1

def rand01():
    num = rand01p()
    while num == rand01p():
        num = rand01p()
    return num

def rand0To3():
    return rand01()*2 + rand01()

def rand1To6():
    num = rand0To3() * 4 + rand0To3()
    while num > 11:
        num = rand0To3() * + rand0To3()
    return num % 6 + 1


def rand1ToM(m):
    return int(random.random() * m) + 1

def rand1ToN(n, m):
    def getMSysNum(value, m):
        res = []
        while value != 0:
            res.append(value % m)
            value //= m
        return res[::-1]

    def getRandMSysNumLessN(nMSys, m):
        res = []
        lastEqual = True
        index = 0
        while index < len(nMSys):
            res.append(rand1ToM(m) - 1)
            if lastEqual:
                if res[-1] > nMSys[index]:
                    index = 0
                    res = []
                    continue
                else:
                    lastEqual = True if res[-1] == nMSys[index] else False
            index += 1
        return res

    def getNumFromMSysNum(mSysNum, m):
        res = 0
        for i in range(len(mSysNum)):
            res = res * m + mSysNum[i]
        return res

    nMSys = getMSysNum(n-1, m)
    randNum = getRandMSysNumLessN(nMSys, m)
    return getNumFromMSysNum(randNum, m) + 1


#一行代码求两个数的最大公约数
def gcd(m, n):
    return m if n == 0 else gcd(n, m % n)


#有关阶乘的两个问题
def zeroNum1(num):
    if num <= 0:
        return 0
    res = 0
    for i in range(5, num+1, 5):
        cur = i
        while cur % 5 == 0:
            res += 1
            cur //= 5
    return res


def zeroNum2(num):
    if num <= 0:
        return 0
    res = 0
    while num != 0:
        res += num // 5
        num //= 5
    return res

def rightOne1(num):
    if num < 1:
        return -1
    res = 0
    while num != 0:
        num >>= 1
        res += num
    return res

def rightOne2(num):
    if num < 1:
        return -1
    res = 0
    tmp = num
    while tmp != 0:
        if tmp & 1 != 0:
            res += 1
        tmp >>= 1
    return num - res


#判断一个点是否在矩形内部
def isInside(x1, y1, x2, y2, x3, y3, x4, y4, x, y):
    def isInside(x1, y1, x4, y4, x, y):
        if x <= x1 or x >= x4 or y <= y4 or y >= y1:
            return False
        return True

    if y1 == y2:
        return isInside(x1, y1, x4, y4, x, y)
    a = abs(y4 - y3)
    b = abs(x4 - x3)
    c = math.sqrt(a*a + b*b)
    sin = a / c
    cos = b / c
    x1R = x1*cos + y1*sin
    y1R = y1*cos - x1*sin
    x4R = x4*cos + y4*sin
    y4R = y4*cos - x4*sin
    xR = x*cos + y*sin
    yR = y*cos - x*sin
    return isInside(x1R, y1R, x4R, y4R, xR, yR)


#判断一个点是否在三角形内部
def isInside1(x1, y1, x2, y2, x3, y3, x, y):
    def getSideLength(x1, y1, x2, y2):
        a = abs(x2 - x1)
        b = abs(y2 - y1)
        return math.sqrt(a*a + b*b)

    def getArea(x1, y1, x2, y2, x3, y3):
        a = getSideLength(x1, y1, x2, y2)
        b = getSideLength(x1, y1, x3, y3)
        c = getSideLength(x2, y2, x3, y3)
        p = (a + b + c) / 2
        return math.sqrt(p * (p-a) * (p-b) * (p-c))

    area1 = getArea(x1, y1, x2, y2, x, y)
    area2 = getArea(x1, y1, x3, y3, x, y)
    area3 = getArea(x2, y2, x3, y3, x, y)
    allArea = getArea(x1, y1, x2, y2, x3, y3)
    return (area1 + area2 + area3) <= allArea


def isInside2(x1, y1, x2, y2, x3, y3, x, y):
    def crossProduct(x1, y1, x2, y2):
        return x1 * y2 - x2 * y1

    if crossProduct(x3-x1, y3-y1, x2-x1, y2-y1) >= 0:
        x2, x3 = x3, x2
        y2, y3 = y3, y2
    if crossProduct(x2-x1, y2-y1, x-x1, y-y1) < 0:
        return False
    if crossProduct(x3-x2, y3-y2, x-x2, y-y2) < 0:
        return False
    if crossProduct(x1-x3, y1-y3, x-x3, y-y3) < 0:
        return False
    return True


#折纸问题
def printAllFolds(N):
    def printProcess(i, N, isDown):
        if i > N:
            return
        printProcess(i+1, N, True)
        print("Down" if isDown else "Up", end=' ')
        printProcess(i+1, N, False)

    printProcess(1, N, True)


#蓄水池算法
def getKNumRand(k, max):
    def rand(max):
        return int(random.random() * max) + 1

    if k < 1 or max < 1:
        return None
    res = []
    for i in range(k):
        res.append(i+1)
    for i in range(k+1, max+1):
        if rand(i) <= k:
            res[rand(k)-1] = i
    return res


#设计有setAll功能的哈希表
class MyValue:
    def __init__(self, value, time):
        self.__value = value
        self.__time = time

    def getValue(self):
        return self.__value

    def getTime(self):
        return self.__time


class MyHashMap:
    def __init__(self):
        self.map = {}
        self.time = 0
        self.setAll = MyValue(None, -1)

    def containsKey(self, key):
        return key in self.map

    def put(self, key, value):
        self.map[key] = MyValue(value, self.time)
        self.time += 1

    def setAll_(self, value):
        self.setAll = MyValue(value, self.time)
        self.time += 1

    def get(self, key):
        if self.containsKey(key):
            if self.map[key].getTime() < self.setAll.getTime():
                return self.setAll.getValue()
            else:
                return self.map[key].getValue()
        else:
            return None


#最大的leftMax与rightMax之差的绝对值
def maxABS1(arr):
    res = 0
    for i in range(len(arr)-1):   #重要
        leftMax = -sys.maxsize
        for j in range(i+1):
            leftMax = max(arr[j], leftMax)
        rightMax = -sys.maxsize
        for k in range(i+1, len(arr)):
            rightMax = max(arr[k], rightMax)
        res = max(abs(rightMax-leftMax), res)
    return res


def maxABS2(arr):
    res = 0
    lHelp = []
    rHelp = []
    lHelp.append(arr[0])
    rHelp.append(arr[-1])
    for i in range(1, len(arr)):
        lHelp.append(arr[i] if arr[i] > lHelp[-1] else lHelp[-1])
    for i in range(len(arr)-2, -1, -1):
        rHelp.append(arr[i] if arr[i] > rHelp[-1] else rHelp[-1])
    rHelp = rHelp[::-1]
    for i in range(len(arr)):
        res = max(abs(lHelp[i] - rHelp[i]), res)
    return res


def maxABS3(arr):
    maxNum = arr[0]
    for i in range(1, len(arr)):
        maxNum = max(arr[i], maxNum)
    return max(abs(maxNum - arr[0]), abs(maxNum - arr[-1]))


#设计可以变更的缓存结构
class Node:
    def __init__(self, value):
        self.value = value
        self.pre = None
        self.next = None

class DoubleLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def addNode(self, newNode):
        if newNode == None:
            return
        if self.head == None:
            self.head = newNode
            self.tail = newNode
        else:
            self.tail.next = newNode
            newNode.pre = self.tail
            self.tail = newNode

    def moveNodeToTail(self, node):
        if node == None:
            return
        if node == self.tail:
            return
        if node == self.head:
            self.head = node.next
            self.head.pre = None
        else:
            node.pre.next = node.next
            node.next.pre = node.pre
        self.tail.next = node
        node.next = None
        node.pre = self.tail
        self.tail = node

    def removeHead(self):
        if self.head == None:
            return None
        res = self.head
        if self.head == self.tail:
            self.head == None
            self.tail == None
        else:
            self.head = res.next
            self.head.pre = None
            res.next = None
        return res


class MyCache:
    def __init__(self, capacity):
        self.nodeKeyMap = {}
        self.keyNodeMap = {}
        self.capacity = capacity
        self.nodeList = DoubleLinkedList()

    def get(self, key):
        if key in self.keyNodeMap:
            res = self.keyNodeMap[key]
            self.nodeList.moveNodeToTail(res)
            return res.value
        return None

    def set(self, key, value):
        if key in self.keyNodeMap:
            node = self.keyNodeMap[key]
            node.value = value
            self.nodeList.moveNodeToTail(node)
        else:
            node = Node(value)
            self.keyNodeMap[key] = node
            self.nodeKeyMap[node] = key
            self.nodeList.addNode(node)
            if len(self.keyNodeMap) == self.capacity + 1:
                self.removeMostUnusedCache()

    def removeMostUnusedCache(self):
        node = self.nodeList.removeHead()
        key = self.nodeKeyMap[node]
        del self.keyNodeMap[key]
        del self.nodeKeyMap[node]


#设计RandomPool结构
class Pool:
    def __init__(self):
        self.keyIndexMap = {}
        self.indexKeyMap = {}
        self.index = 0

    def insert(self, key):
        self.keyIndexMap[key] = self.index
        self.indexKeyMap[self.index] = key
        self.index += 1

    def delete(self, key):
        if key in self.keyIndexMap:
            index = self.keyIndexMap[key]
            lastKey = self.indexKeyMap[self.index-1]
            self.indexKeyMap[index] = lastKey
            del self.keyIndexMap[lastKey]
            del self.indexKeyMap[self.index-1]
            self.index -= 1

    def getRandom(self):
        if self.index == 0:
            return None
        index = int(random.random() * self.index)
        return self.indexKeyMap[index]


#调整[0,x)区间上的数出现的概率
def randXPowerK(k):
    if k < 1:
        return 0
    res = -1
    for i in range(k):
        res = max(res, random.random())
    return res


#路径数组变为统计数组
def pathsToNums(paths):
    def pathsToDistance(paths):
        cap = -1
        for i in range(len(paths)):
            if paths[i] == i:
                cap = i
            elif paths[i] > -1:
                curI = paths[i]
                preI = i
                paths[i] = -1
                while paths[curI] != curI:
                    if paths[curI] > -1:
                        next = paths[curI]
                        paths[curI] = preI
                        preI = curI
                        curI = next
                    else:
                        break
                value = 0 if paths[curI] == curI else paths[curI]
                while paths[preI] != -1:
                    index = paths[preI]
                    paths[preI] = value - 1 
                    value -= 1
                    preI = index
                paths[preI] = value - 1
        paths[cap] = 0

        
    def distanceToNums(disArr):
        for i in range(len(disArr)):
            index = disArr[i]
            if index < 0:
                disArr[i] = 0
                index = -index
                while disArr[index] < 0:
                    tmp = disArr[index]
                    disArr[index] = 1
                    index = -tmp
                disArr[index] += 1
        disArr[0] = 1


    if paths == None or len(paths) == 0:
        return
    pathsToDistance(paths)
    distanceToNums(paths)
    return paths


#正数数组的最小不可组成和
def unformedSum1(arr):
    def process(arr, i, sum, set1):
        if i == len(arr):
            set1.add(sum)
            return
        process(arr, i+1, sum, set1)
        process(arr, i+1, sum + arr[i], set1)


    if arr == None or len(arr) == 0:
        return 1
    set1 = set()
    process(arr, 0, 0, set1)
    min1 = sys.maxsize
    for i in range(len(arr)):
        min1 = min(arr[i], min1)
    for i in range(min1+1, sys.maxsize):
        if i not in set1:
            return i
    return 0


def unformedSum2(arr):
    if arr == None or len(arr) == 0:
        return 1
    maxSum = 0
    minSum = sys.maxsize
    for i in range(len(arr)):
        minSum = min(arr[i], minSum)
        maxSum += arr[i]
    dp = [False for i in range(maxSum+1)]
    dp[0] = True
    for i in range(len(arr)):
        for j in range(maxSum,arr[i]-1, -1):
            if dp[j-arr[i]]:
                dp[j] = True
    for i in range(minSum, len(dp)):
        if not dp[i]:
            return i
    return maxSum + 1 


def unformedSum3(arr):
    if arr == None or len(arr) == 0:
        return 1
    arr.sort()
    rang = 0
    for i in range(len(arr)):
        if arr[i] <= rang+1:
            rang += arr[i]
        else:
            break
    return rang + 1


#一种字符串和数字的对应关系
def getString(chs, n):
    if chs == None or len(chs) == 0 or n < 1:
        return ""
    k = len(chs)
    cur = 1
    length = 0
    while n - cur >= 0:
        n -= cur
        cur *= k
        length += 1
    res = [0 for i in range(length)]
    for i in range(length):
        cur //= k
        nCur = n // cur
        res[i] = chs[nCur]
        n %= cur
    return ''.join(res)

def getNum(chs, str1):
    def getNthFromChar(chs, ch):
        for i in range(len(chs)):
            if chs[i] == ch:
                return i+1
        return -1

    if chs == None or len(chs) == 0 or str1 == None or len(str1) == 0:
        return 0
    res = 0
    k = len(chs)
    for i in range(len(str1)):
        res = res * k + getNthFromChar(chs, str1[i])
    return res


#1到n中1出现的次数
def oneNums1(num):
    if num < 1:
        return 0
    res = 0
    for i in range(1, num+1):
        cur = i
        tmp = 0
        while cur != 0:
            if cur % 10 == 1:
                tmp += 1
            cur //= 10
        res += tmp
    return res


def oneNums2(num):
    def getLength(num):
        res = 0
        while num != 0:
            res += 1
            num //= 10
        return res

    if num < 1:
        return 0
    length = getLength(num)
    base = int(math.pow(10, length-1))
    first = num // base
    firstOneNum = num % base + 1 if first == 1 else base
    otherOneNum = first * (length-1) * (base // 10)
    return firstOneNum + otherOneNum + oneNums2(num % base)


#从N个数中等概率打印M个数
def printRandM(arr, m):
    if arr == None or len(arr) == 0 or len(arr) < m or m < 1:
        return
    res = 0
    n = len(arr)
    while res < m:
        index = int(random.random() * n)
        print(arr[index], end=' ')
        arr[index], arr[n-1] = arr[n-1], arr[index]
        n -= 1
        res += 1


#判断一个数是否是回文数
def isPlindrome(num):
    if num == -(1 << 31):
        return False
    num = abs(num)
    base = 1
    while base <= num:
        base *= 10
    base //= 10
    while num != 0:
        if num % 10 != num // base:
            return False
        num = num % base // 10
        base //= 100
    return True


#在有序旋转数组中找到最小值
def getMin(arr):
    if arr == None or len(arr) == 0:
        return 0
    left = 0
    right = len(arr) - 1
    while left < right:
        if left == right - 1:
            break
        if arr[left] < arr[right]:
            return arr[left]
        mid = (left + right) // 2
        if arr[left] > arr[mid]:
            right = mid
        elif arr[mid] > arr[right]:
            left = mid
        else:
            while left != mid:
                if arr[left] == arr[mid]:
                    left += 1
                elif arr[left] > arr[mid]:
                    right = mid
                    break
                else:
                    return arr[left]
    return min(arr[left], arr[right])


#在有序旋转数组中找到一个数
def isContains(arr, num):
    if arr == None or len(arr) == 0:
        return False
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == num:
            return True
        if arr[left] == arr[mid] == arr[right]:
            while left < mid and arr[left] == arr[mid]:
                left += 1
            if left == mid:
                left = mid + 1
                continue
        if arr[left] != arr[mid]:
            if arr[left] < arr[mid]:
                if num < arr[mid] and num >= arr[left]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if num > arr[mid] and num <= arr[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        else:
            if arr[mid] < arr[right]:
                if num > arr[mid] and num <= arr[right]:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                if num >= arr[left] and num < arr[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
    return False


#数字的英文表达和中文表达
def num1To19(num):
    if num < 1 or num > 19:
        return ""
    names = ["One","Two","Three","Four","Five","Six","Seven","Eight","Nine",\
            "Ten","Eleven","Twelve","Thirteen","Fourteen","Fifteen",\
            "Sixteen","Seventeen","Eighteen","Nineteen"]
    return names[num-1]

def num1To99(num):
    if num < 1 or num > 99:
        return ""
    if num >= 1 and num <= 19:
        return num1To19(num)
    high = num // 10
    tyNames = ["Twenty","Thirty","Forty","Fifty","Sixty","Seventy","Eighty","Ninety"]
    return tyNames[high-2] + " " + num1To19(num % 10)

def num1To999(num):
    if num < 1 or num > 999:
        return ""
    if num >= 1 and num <= 99:
        return num1To99(num)
    return num1To19(num // 100) + " Hundred " + num1To99(num % 100)

def getNumEngExp(num):
    if num == 0:
        return "Zero"
    res = ""
    if num < 0:
        res = "Negetive, "
    if num == -(1 << 31):    #防止取绝对值时溢出
        res += "Two Billion, "
        num %= -2000000000
    num = abs(num)
    high = 1000000000
    names = ["Billion","Million","Thousand",""]
    index = 0
    while num != 0:
        cur = num // high
        num %= high
        if cur != 0:
            res += num1To999(cur) + " " + names[index]
            res += ", " if num != 0 else ""
        index += 1
        high //= 1000
    return res


def num1To9ByChinese(num):
    if num < 1 or num > 9:
        return ""
    names = ["一","二","三","四","五","六","七","八","九"]
    return names[num-1]

def num1To99ByChinese(num, hasBai):
    if num < 1 or num > 99:
        return ""
    if num < 10:
        return num1To9ByChinese(num)
    shi = num // 10
    if shi == 1 and not hasBai:
        return "十" + num1To9ByChinese(num % 10)
    else:
        return num1To9ByChinese(shi) + "十" + num1To9ByChinese(num % 10)

def num1To999ByChinese(num):
    if num < 1 or num > 999:
        return ""
    if num < 100:
        return num1To99ByChinese(num, False)
    res = num1To9ByChinese(num // 100) + "百"
    rest = num % 100
    if rest == 0:
        return res
    elif rest >= 10:
        res += num1To99ByChinese(rest, True)
    else:
        res += "零" + num1To9ByChinese(rest)
    return res

def num1To9999ByChinese(num):
    if num < 1 or num > 9999:
        return ""
    if num < 1000:
        return num1To999ByChinese(num)
    res = num1To9ByChinese(num // 1000) + "千"
    rest = num % 1000
    if rest == 0:
        return res
    elif rest >= 100:
        res += num1To999ByChinese(rest)
    else:
        res += "零" + num1To99ByChinese(rest, False)
    return res

def num1To99999999ByChinese(num):
    if num < 1 or num > 99999999:
        return ""
    if num < 10000:
        return num1To9999ByChinese(num)
    wan = num // 10000
    rest = num % 10000
    res = num1To9999ByChinese(wan) + "万"
    if rest == 0:
        return res
    elif rest > 1000:
        res += num1To9999ByChinese(rest)
    else:
        res += "零" + num1To999ByChinese(rest)
    return res

def getNumChineseExp(num):
    if num == 0:
        return "零"
    res = ""
    if num < 0:
        res = "负"
    num = abs(num)
    yi = num // 100000000
    rest = num % 100000000
    if yi == 0:
        return res + num1To99999999ByChinese(rest)
    res += num1To99999999ByChinese(yi) + "亿"
    if rest == 0:
        return res
    elif rest > 10000000:
        return res + num1To99999999ByChinese(rest)
    else:
        return res + "零" + num1To99999999ByChinese(rest)


#分糖果问题
def candy1(arr):
    def nextMinIndex(arr, start):
        for i in range(start, len(arr)-1):
            if arr[i+1] >= arr[i]:
                return i
        return len(arr) - 1

    def rightCands(left, right):
        n = right - left + 1
        return n * (n+1) // 2


    if arr == None or len(arr) == 0:
        return 0
    index = nextMinIndex(arr, 0)
    res = rightCands(0, index)
    index += 1
    lbase = 1
    while index != len(arr):
        if arr[index] > arr[index-1]:
            lbase += 1
            res += lbase
            index += 1
        elif arr[index] < arr[index-1]:
            next = nextMinIndex(arr, index-1)
            res += rightCands(index-1, next)
            rbase = next - index + 2
            res -= rbase if rbase < lbase else lbase
            lbase = 1
            index = next + 1
        else:
            res += 1
            lbase = 1
            index += 1
    return res


def candy2(arr):
    def nextMinIndex(arr, start):
        for i in range(start, len(arr)-1):
            if arr[i] < arr[i+1]:
                return i
        return len(arr) - 1

    def rightCandsAndBase(arr, left, right):
        res = 1
        base = 1
        for i in range(right-1, left-1, -1):
            if arr[i] == arr[i+1]:
                res += base
            else:
                base += 1
                res += base
        return res, base

    if arr == None or len(arr) == 0:
        return 0
    index = nextMinIndex(arr, 0)
    res, s = rightCandsAndBase(arr, 0, index)
    index += 1
    lbase = 1
    same = 1
    while index != len(arr):
        if arr[index] > arr[index-1]:
            lbase += 1
            res += lbase
            index += 1
        elif arr[index] == arr[index-1]:
            res += lbase
            same += 1
            index += 1
        else:
            next = nextMinIndex(arr, index-1)
            num, rbase = rightCandsAndBase(arr, index-1, next)
            if rbase < lbase:
                res += num - rbase
            else:
                res += num - rbase + rbase * same - same * lbase
            index = next + 1
            same = 1
            lbase = 1
    return res


#一种消息接受并打印的结构设计
class Node2:
    def __init__(self, value):
        self.value = value
        self.next = None

class MessageBox:
    def __init__(self):
        self.headMap = {}
        self.tailMap = {}
        self.lastPrint = 0

    def receive(self, value):
        node = Node(value)
        self.headMap[value] = node
        self.tailMap[value] = node
        if value + 1 in self.headMap:
            node.next = self.headMap[value+1]
            del self.headMap[value+1]
            del self.tailMap[value]
        if value - 1 in self.tailMap:
            self.tailMap[value-1].next = node
            del self.tailMap[value-1]
            del self.headMap[value]
        if self.lastPrint + 1 in self.headMap:
            self.printValue()

    def printValue(self):
        head = self.headMap[self.lastPrint+1]
        del self.headMap[self.lastPrint+1]
        while head != None:
            print(head.value, end=' ')
            head = head.next
            self.lastPrint += 1
        del self.tailMap[self.lastPrint]


#设计一个没有扩容负担的堆结构
class HeapNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.parent = None

class MyHeap:
    def __init__(self, comp):
        self.head = None    #堆头节点
        self.last = None    #堆尾节点
        self.size = 0    #当前堆的大小
        self.comp = comp    #基于比较器决定是大根堆还是小根堆

    def getHead(self):
        return self.head.value if self.head != None else None

    def getSize(self):
        return self.size

    def isEmpty(self):
        return True if self.size == 0 else False

    #添加一个新节点到堆中
    def add(self, value):
        newNode = HeapNode(value)
        if self.size == 0:
            self.head = newNode
            self.last = newNode
            self.size = 1
            return
        node = self.last
        parent = node.parent   
        #找到尾节点的下一个位置并插入新节点
        while parent != None and parent.left != node:
            node = parent
            parent = node.parent
        if parent == None:
            nodeToAdd = self.mostLeft(self.head)
            nodeToAdd.left = newNode
            newNode.parent = nodeToAdd
        elif parent.right == None:
            parent.right = newNode
            newNode.parent = parent
        else:
            nodeToAdd = self.mostLeft(parent.right)
            nodeToAdd.left = newNode
            newNode.parent = nodeToAdd
        self.last = newNode
        #建堆过程及其调整
        self.heapInsertModify()
        self.size += 1

    def heapInsertModify(self):
        node = self.last
        parent = node.parent
        if parent != None and self.comp(parent.value, node.value):
            self.last = node.parent
        while parent != None and self.comp(parent.value, node.value):
            self.swapClosedTwoNode(node, parent)
            parent = node.parent
        if self.head.parent != None:
            self.head = self.head.parent

    def swapClosedTwoNode(self, node, parent):
        if node == None or parent == None:
            return 
        parentParent = parent.parent
        parentLeft = parent.left
        parentRight = parent.right
        nodeLeft = node.left
        nodeRight = node.right
        node.parent = parentParent
        if parentParent != None:
            if parentParent.left == parent:
                parentParent.left = node
            else:
                parentParent.right = node
        parent.parent = node
        if nodeLeft != None:
            nodeLeft.parent = parent
        if nodeRight != None:
            nodeRight.parent = parent
        if node == parentLeft:
            node.left = parent
            node.right = parentRight
            if parentRight != None:
                parentRight.parent = node
        else:
            node.left = parentLeft
            node.right = parent
            if parentLeft != None:
                parentLeft.parent = node
        parent.left = nodeLeft
        parent.right = nodeRight

    def mostLeft(self,node):
        while node.left != None:
            node = node.left
        return node

    def mostRight(self, node):
        while node.right != None:
            node = node.right
        return node

    def popHead(self):
        if self.size == 0:
            return None
        res = self.head
        if self.size == 1:
            self.head = None
            self.last = None
            self.size = 0
            return res.value
        oldLast = self.popLastAndSetPreviousLast()  #返回尾节点并更新尾节点
        #如果弹出尾节点，堆的大小等于1的处理
        if self.size == 1:
            self.head = oldLast
            self.last = oldLast
            return res.value
        #如果弹出尾节点，堆的大小大于1的处理
        headLeft = res.left
        headRight = res.right
        oldLast.left = headLeft
        if headLeft != None:
            headLeft.parent = oldLast
        oldLast.right = headRight
        if headRight != None:
            headRight.parent = oldLast
        res.left = None
        res.right = None
        self.head = oldLast
        #堆heapify过程
        self.heapify(self.head)
        return res.value

    def heapify(self, node):
        left = node.left
        right = node.right
        most = node
        while left != None:
            if left != None and self.comp(most.value, left.value):
                most = left
            if right != None and self.comp(most.value, right.value):
                most = right
            if most == node:
                break
            else:
                self.swapClosedTwoNode(most, node)
            left = node.left
            right = node.right
            most = node
        if node.parent == self.last:
            self.last = node
        while node.parent != None:
            node = node.parent
        self.head = node

    def popLastAndSetPreviousLast(self):
        node = self.last
        parent = node.parent  #以下是寻找尾节点的上一个节点
        while parent != None and parent.right != node:
            node = parent
            parent = node.parent
        if parent == None:
            node = self.last
            parent = node.parent
            node.parent = None
            if parent != None:
                parent.left = None
            self.last = self.mostRight(self.head)
        else:
            newLast = self.mostRight(parent.left)
            node = self.last
            parent = node.parent
            node.parent = None
            if parent.left == node:
                parent.left = None
            else:
                parent.right = None
            self.last = newLast
        self.size -= 1
        return node


#随时找到数据流的中位数
def minHeapComparator(a, b):
    if a <= b:
        return False
    if a > b:
        return True

def maxHeapComparator(a, b):
    if a >= b:
        return False
    else:
        return True


class MedianHolder:
    def __init__(self):
        self.minHeap = MyHeap(minHeapComparator)
        self.maxHeap = MyHeap(maxHeapComparator)

    def addNum(self, num):
        if self.maxHeap.isEmpty():
            self.maxHeap.add(num)
            return
        if self.maxHeap.getHead() >= num:
            self.maxHeap.add(num)
        else:
            if self.minHeap.isEmpty():
                self.minHeap.add(num)
            elif self.minHeap.getHead() > num:
                self.maxHeap.add(num)
            else:
                self.minHeap.add(num)
        self.modifyTwoHeapSize()

    def modifyTwoHeapSize(self):
        if self.minHeap.getSize() == self.maxHeap.getSize() + 2:
            self.maxHeap.add(self.minHeap.popHead())
        if self.maxHeap.getSize() == self.minHeap.getSize() + 2:
            self.minHeap.add(self.maxHeap.popHead())

    def getMedian(self):
        maxHeapSize = self.maxHeap.getSize()
        minHeapSize = self.minHeap.getSize()
        if maxHeapSize + minHeapSize == 0:
            return None
        if (maxHeapSize + minHeapSize) & 1 == 0:
            return (self.maxHeap.getHead() + self.minHeap.getHead()) / 2
        else:
            if maxHeapSize > minHeapSize:
                return self.maxHeap.getHead()
            else:
                return self.minHeap.getHead()


#在两个长度相等的排序数组中找到上中位数
def getUpMedian(arr1, arr2):
    if arr1 == None or arr2 == None or len(arr1) != len(arr2):
        raise Exception("Your arr is invalid!")
    start1 = 0
    end1 = len(arr1) - 1
    start2 = 0
    end2 = len(arr2) - 1
    while start1 < end1:
        mid1 = (start1 + end1) // 2
        mid2 = (start2 + end2) // 2
        offset = (end1 - start1 + 1) & 1 ^ 1
        if arr1[mid1] == arr2[mid2]:
            return arr1[mid1]
        elif arr1[mid1] > arr2[mid2]:
            end1 = mid1
            start2 = mid2 + offset
        else:
            start1 = mid1 + offset
            end2 = mid2
    return min(arr1[start1], arr2[start2])


#在两个排序数组中找到第Ｋ小的数
def findKthNum(arr1, arr2, k):
    def getUpMedian(a1, s1, e1, a2, s2, e2):
        while s1 < e1:
            mid1 = (e1 + s1) // 2
            mid2 = (e2 + s2) // 2
            offset = (e1 - s1 + 1) & 1 ^ 1
            if arr1[mid1] == arr2[mid2]:
                return arr1[mid1]
            elif arr1[mid1] > arr2[mid2]:
                e1 = mid1
                s2 = mid2 + offset
            else:
                s1 = mid1 + offset
                e2 = mid2
        return min(arr1[s1], arr2[s2])

    if arr1 == None or arr2 == None:
        raise Exception("Your arr is invalid!")
    if k < 1 or k > (len(arr1) + len(arr2)):
        raise Exception("K is invalid!")
    longs = arr1 if len(arr1) > len(arr2) else arr2
    shorts = arr1 if len(arr1) <= len(arr2) else arr2
    l = len(longs)
    s = len(shorts)
    if k <= s:
        return getUpMedian(shorts, 0, k-1, longs, 0, k-1)
    if k > l:
        if longs[k-s-1] >= shorts[-1]:
            return longs[k-s-1]
        if shorts[k-l-1] >= longs[-1]:
            return shorts[k-l-1]
        return getUpMedian(longs, k-s, l-1, shorts, k-l, s-1)
    if longs[k-s-1] >= shorts[-1]:
        return longs[k-s-1]
    print(222)
    return getUpMedian(longs, k-s, k-1, shorts, 0, s-1)


#两个有序数组间相加和的TopK问题
class Heap:
    def __init__(self, row, col, value):
        self.row = row
        self.col = col
        self.value = value

def getTopKSum(arr1, arr2, k):
    def heapInsert(heap, row, col, data, i):
        node = Heap(row, col, data)
        heap[i] = node
        parent = (i-1) // 2
        while parent >= 0 and heap[parent].value < heap[i].value:
            heap[parent], heap[i] = heap[i], heap[parent]
            i = parent
            parent = (i-1) // 2

    def popHead(heap, heapSize):
        res = heap[0]
        heap[0], heap[heapSize-1] = heap[heapSize-1], heap[0]
        heapify(heap, 0, heapSize-1)
        return res

    def heapify(heap, i, heapSize):
        left = 2 * i + 1
        right = 2 * i + 2
        most = i
        while left < heapSize:
            if heap[left].value > heap[i].value:
                most = left
            if right < heapSize and heap[right].value > heap[most].value:
                most = right
            if most == i:
                break
            else:
                heap[most], heap[i] = heap[i], heap[most]
                i = most
                left = 2 * i + 1
                right = 2 * i + 2

    def isContains(row, col, posSet):
        return '_'.join([str(row),str(col)]) in posSet

    def addPosToSet(row, col, posSet):
        posSet.add('_'.join([str(row), str(col)]))



    if arr1 == None or arr2 == None or k < 1 or k > len(arr1) * len(arr2):
        return
    heap = [0 for i in range(k)]
    row = len(arr1) - 1
    col = len(arr2) - 1
    heapSize = 0
    heapInsert(heap, row, col, arr1[row] + arr2[col], heapSize)
    heapSize += 1
    posSet = set()
    count = 0
    res = []
    while count < k:
        cur = popHead(heap, heapSize)
        heapSize -= 1
        res.append(cur.value)
        r = cur.row
        c = cur.col
        if not isContains(r-1,c, posSet):
            heapInsert(heap, r-1, c, arr1[r-1] + arr2[c], heapSize)
            heapSize += 1
            addPosToSet(r-1, c, posSet)
        if not isContains(r, c-1, posSet):
            heapInsert(heap, r, c-1, arr1[r] + arr2[c-1], heapSize)
            heapSize += 1
            addPosToSet(r, c-1, posSet)
        count += 1
    return res


#出现次数的TopK问题
class FreNode:
    def __init__(self, st, times):
        self.str = st
        self.times = times

def printTopKAndRank(strArr, k):
    def heapInsert(heap, i):
        parent = (i - 1) // 2
        while parent >= 0 and heap[parent].times > heap[i].times:
            heap[parent], heap[i] = heap[i], heap[parent]
            i = parent
            parent = (i - 1) // 2

    def heapify(heap, i, heapSize):
        left = 2 * i + 1
        right = 2 * i + 2
        most = i
        while left < heapSize:
            if heap[left].times < heap[i].times:
                most = left
            if right < heapSize and heap[right].times < heap[most].times:
                most = right
            if most == i:
                break
            else:
                heap[most], heap[i] = heap[i], heap[most]
                i = most
                left = 2 * i + 1
                right = 2 * i + 2

    if strArr == None or len(strArr) == 0 or k < 1 or k > len(strArr):
        return
    map = {}
    for element in strArr:
        if element in map:
            map[element] += 1
        else:
            map[element] = 1
    heap = [0 for i in range(k)]
    index = 0
    for key,value in map.items():
        curNode = FreNode(key, value)
        if index != k:
            heap[index] = curNode
            heapInsert(heap, index)
            index += 1
        else:
            if heap[0].times < curNode.times:
                heap[0] = curNode
                heapify(heap, 0, k)
    for i in range(index-1, 0, -1):
        heap[0], heap[i] = heap[i],heap[0]
        heapify(heap,0,i)
    for i in range(index):
        print("No." + str(i+1) + " :" + heap[i].str + " times: " + str(heap[i].times))


class TopKRecord:
    index = 0    #目前堆中的元素个数
    strNodeMap = {}    #记录字符串和node的对应关系
    nodeIndexMap = {}   #记录node在堆中的位置，如果不在堆中则为-1

    def __init__(self, size):
        self.heap = [0 for i in range(size)]

    def add(self, str1):
        preIndex = -1
        curNode = None
        if str1 not in self.strNodeMap:
            curNode = FreNode(str1, 1)
            self.strNodeMap[str1] = curNode
            self.nodeIndexMap[curNode] = -1
        else:
            self.strNodeMap[str1].times += 1
            curNode = self.strNodeMap[str1]
            preIndex = self.nodeIndexMap[curNode]
        if preIndex == -1:
            if self.index == len(self.heap):
                if curNode.times > self.heap[0].times:
                    self.nodeIndexMap[self.heap[0]] = -1
                    self.nodeIndexMap[curNode] = 0
                    self.heap[0] = curNode
                    self.heapify(0, self.index)
            else:
                self.nodeIndexMap[curNode] = self.index
                self.heap[self.index] = curNode
                self.heapInsert(self.index)
                self.index += 1
        else:
            self.heapify(preIndex, self.index)

    def printTopK(self):
        print("TOP:")
        for i in range(self.index):
            print("Str: " + self.heap[i].str + " Times:" + str(self.heap[i].times))

    def heapify(self, i, heapSize):
        left = 2 * i + 1
        right = 2 * i + 2
        smallest = i
        while left < heapSize:
            if self.heap[left].times < self.heap[i].times:
                smallest = left
            if right < heapSize and self.heap[right].times < self.heap[smallest].times:
                smallest = right
            if smallest == i:
                break
            else:
                self.nodeIndexMap[self.heap[i]] = smallest
                self.nodeIndexMap[self.heap[smallest]] = i
                self.heap[i], self.heap[smallest] = self.heap[smallest], self.heap[i]
                i = smallest
                left = 2 * i + 1
                right = 2 * i + 2

    def heapInsert(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self.heap[parent].times > self.heap[i].times:
                self.nodeIndexMap[self.heap[i]] = parent
                self.nodeIndexMap[self.heap[parent]] = i
                self.heap[i], self.heap[parent] = self.heap[parent], self.heap[i]
                i = parent
            else:
                break


#Manacher算法
def manacherString(string):
    res = [0 for i in range(len(string) * 2 + 1)]
    index = 0
    for i in range(len(res)):
        if i & 1 == 0:
            res[i] = "#"
        else:
            res[i] = string[index]
            index += 1
    return res

def maxLcpsLength(str1):
    if str1 == None or len(str1) == 0:
        return 0
    mStr = manacherString(str1)
    help = [0 for i in range(len(mStr))]
    index = -1
    right = -1
    maxLen = 0
    for i in range(len(mStr)):
        if right > i:
            help[i] = min(right - i, 2 * index - i)
        else:
            help[i] = 1
        while i + help[i] < len(mStr) and i - help[i] > -1:
            if mStr[i+help[i]] == mStr[i-help[i]]:
                help[i] += 1
            else:
                break
        if i + help[i] > right:
            right = i + help[i]
            index = i
        maxLen = max(maxLen, help[i])
    return maxLen - 1


def shortestEnd(str1):
    if str1 == None or len(str1) == 0:
        return 0
    mStr = manacherString(str1)
    help = [0 for i in range(len(mStr))]
    index = -1
    right = -1
    maxLen = 0
    maxContainsEnd = 0
    for i in range(len(mStr)):
        if right > i:
            help[i] = min(right-i, help[2*index-i])
        else:
            help[i] = 1
        while i + help[i] < len(mStr) and i - help[i] > -1:
            if mStr[i+help[i]] == mStr[i-help[i]]:
                help[i] += 1
            else:
                break
        if i + help[i] > right:
            right = i + help[i]
            index = i
        if right == len(mStr):
            maxContainsEnd = help[i]
            break
    res = []
    for i in range(len(str1)-maxContainsEnd+1):
        res.append(str1[i])
    return ''.join(res[::-1])


#KMP算法
def getIndexOf(strS, strM):
    def getNextArray(str1):
        if len(str1) == 1:
            return [-1]
        nextArr = [0 for i in range(len(str1))]
        nextArr[0] = -1
        nextArr[1] = 0
        pos = 2
        cn = 0
        while pos < len(str1):
            if str1[pos-1] == str1[cn]:
                nextArr[pos] = cn + 1
                pos += 1
                cn += 1
            elif cn == 0:
                nextArr[pos] = 0
                pos += 1
            else:
                cn = nextArr[cn]
        return nextArr
        
    if strS == None or strM == None or len(strM) < 1 or len(strS) < len(strM):
        return -1
    nextArr = getNextArray(strM)
    si = 0
    mi = 0
    while si < len(strS) and mi < len(strM):
        if strS[si] == strM[mi]:
            si += 1
            mi += 1
        elif mi == 0:
            si += 1
        else:
            mi = nextArr[mi]
    return -1 if mi != len(strM) else si - mi 


#丢棋子问题
def chessQuestion1(nLevel, kChess):
    def process1(nLevel, kChess):
        if nLevel == 0:
            return 0
        if kChess == 1:
            return nLevel
        minTimes = sys.maxsize
        for i in range(1, nLevel+1):
            minTimes = min(minTimes, max(process1(i-1, kChess-1), process1(nLevel-i, kChess)))
        return minTimes + 1

    if nLevel < 1 or kChess <1:
        return 0
    if kChess == 1:
        return nLevel
    return process1(nLevel, kChess)


def chessQuestion2(nLevel, kChess):
    if nLevel < 1 or kChess < 1:
        return 0
    if kChess == 1:
        return nLevel
    dp = [[0 for i in range(nLevel+1)] for j in range(kChess+1)]
    for i in range(nLevel+1):
        dp[1][i] = i
    for i in range(2, kChess+1):
        for j in range(1, nLevel+1):
            dp[i][j] = sys.maxsize
            for k in range(1, j+1):
                dp[i][j] = min(dp[i][j], max(dp[i-1][k-1], dp[i][j-k]))
            dp[i][j] += 1
    return dp[-1][-1]
            

def chessQuestion3(nLevel, kChess):
    if nLevel < 1 or kChess < 1:
        return 0
    if kChess == 1:
        return nLevel
    preArr = [0 for i in range(nLevel+1)]
    curArr = [0 for i in range(nLevel+1)]
    for i in range(nLevel+1):
        curArr[i] = i
    for i in range(2, kChess+1):
        preArr = copy.copy(curArr)
        curArr[2] = 0
        for j in range(1, nLevel+1):
            minTimes = sys.maxsize
            for k in range(1, j+1):
                minTimes = min(minTimes, max(preArr[k-1], curArr[j-k]))
            curArr[j] = minTimes + 1
    return curArr[-1]


def chessQuestion4(nLevel, kChess):
    if nLevel < 1 or kChess < 1:
        return 0
    if kChess == 1:
        return nLevel
    dp = [[0 for i in range(kChess+1)] for j in range(nLevel+1)]
    cands = [0 for i in range(kChess+1)]
    for i in range(nLevel+1):
        dp[i][1] = i
    for i in range(1, kChess+1):
        dp[1][i] = 1
        cands[i] = 1
    for i in range(2, nLevel+1):
        for j in range(kChess, 1, -1):
            minEnum = cands[j]
            maxEnum = i if j == kChess else cands[j+1]
            minTimes = sys.maxsize
            for k in range(minEnum, maxEnum+1):
                cur = max(dp[k-1][j-1], dp[i-k][j])
                if cur <= minTimes:
                    minTimes = cur
                    cands[j] = k
            dp[i][j] = minTimes + 1
    return dp[-1][-1]


#画匠问题
def painterQuestion1(arr, num):
    if arr == None or len(arr) == 0 or num < 1:
        raise Exception("Error!")
    if len(arr) == 1:
        return arr[0]
    if len(arr) <= num:
        minTime = arr[0]
        for i in range(len(1, arr)):
            minTime = max(minTime, arr[i])
        return minTime
    sumArr = [0 for i in range(len(arr))]
    map = [0 for i in range(len(arr))]
    sumArr[0] = arr[0]
    map[0] = arr[0]
    for i in range(1, len(arr)):
        sumArr[i] = sumArr[i-1] + arr[i]
        map[i] = sumArr[i]
    for i in range(1, num):
        for j in range(len(arr)-1, i-1, -1):
            minTime = sys.maxsize
            for k in range(i-1, j+1):
                minTime = min(minTime, max(map[k], sumArr[j]-sumArr[k]))
            map[j] = minTime
    return map[-1]


def painterQuestion2(arr, num):
    if arr == None or len(arr) == 0 or num < 1:
        raise Exception("Error!")
    if len(arr) == 1:
        return arr[0]
    if len(arr) <= num:
        minTime = arr[0]
        for i in range(1, len(arr)):
            minTime = max(minTime, arr[i])
        return minTime
    sumArr = [0 for i in range(len(arr))]
    map = [0 for i in range(len(arr))]
    sumArr[0] = arr[0]
    map[0] = arr[0]
    for i in range(1, len(arr)):
        sumArr[i] = sumArr[i-1] + arr[i]
        map[i] = sumArr[i]
    cands = [0 for i in range(len(arr))]
    for i in range(1, num):
        for j in range(len(arr)-1, i-1, -1):
            minEnum = cands[j]
            maxEnum = j if j == len(arr)-1 else cands[j+1]
            minTime = sys.maxsize
            for k in range(minEnum, maxEnum+1):
                cur = max(map[k], sumArr[j] - sumArr[k])
                if cur < minTime:
                    minTime = cur
                    cands[j] = k
            map[j] = minTime
    return map[-1]


def painterQuestion3(arr, num):
    def getNeedNum(arr, limit):
        res = 1
        sum = 0
        for i in range(len(arr)):
            if arr[i] > limit:
                return sys.maxsize
            sum += arr[i]
            if sum > limit:
                res += 1
                sum = arr[i]
        return res

    if arr == None or len(arr) == 0 or num < 1:
        raise Exception("Error!")
    if len(arr) == 1:
        return arr[0]
    if len(arr) <= num:
        minTime = arr[0]
        for i in range(1, len(arr)):
            minTime = max(minTime, arr[i])
        return minTime
    minSum = 0
    maxSum = 0
    for i in range(len(arr)):
        maxSum += arr[i]
    while minSum != maxSum - 1:
        mid = (minSum + maxSum) // 2
        if getNeedNum(arr, mid) > num:
            minSum = mid
        else:
            maxSum = mid
    return maxSum


#邮局选址问题
def siteSelectionQuestion1(arr, num):
    if arr == None or len(arr) == 0 or num < 1 or len(arr) < num:
        return 0
    w = [[0 for i in range(len(arr))] for j in range(len(arr))]
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            w[i][j] = w[i][j-1] + arr[j] - arr[(i+j) // 2]
    dp = [w[0][i] for i in range(len(arr))]
    for i in range(1, num):
        for j in range(len(arr)-1, i-1, -1):
            minDistance = sys.maxsize
            for k in range(i-1, j):
                minDistance = min(minDistance, max(dp[k], w[k+1][j]))
            dp[j] = minDistance
    return dp[-1]


def siteSelectionQuestion2(arr, num):
    if arr == None or len(arr) == 0 or num < 1 or len(arr) < num:
        return 0
    w = [[0 for i in range(len(arr))] for j in range(len(arr))]
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            w[i][j] = w[i][j-1] + arr[j] - arr[(i+j)//2]
    dp = [w[0][i] for i in range(len(arr))]
    cands = [0 for i in range(len(arr))]
    for i in range(1, num):
        for j in range(len(arr)-1, i-1, -1):
            minEnum = cands[j]
            maxEnum = j-1 if j == len(arr)-1 else cands[j+1]
            minDistance = sys.maxsize
            for k in range(minEnum, maxEnum+1):
                cur = max(dp[k], w[k+1][j])
                if cur <= minDistance:
                    minDistance = cur
                    cands[j] = k
            dp[j] = minDistance
    return dp[-1]






print(gcd(10, 5))
print(zeroNum2(1000))
print(rightOne2(10000100000))
print(rand1To7())
print(rand1To6())
print(rand1ToN(10, 5))
print(isInside(1, 3, 4, 6, 4, 0, 7, 3, 6, 1))
print(isInside2(1, 1, 4, 5, 6, 1, 5, 4))
printAllFolds(2)
print()
print(getKNumRand(1, 3))
print(getKNumRand(3, 10))
class1 = MyHashMap()
class1.put(1,3)
class1.put(2,4)
class1.put(3,5)
print(class1.get(1))
class1.setAll_(8)
print(class1.get(1))
class1.put(6,7)
print(class1.get(6))
print(maxABS3([2,7,3,1,1]))
cache = MyCache(2)
cache.set(1,'a')
cache.set(2,'b')
cache.get(1)
cache.set(3,'c')
print(cache.keyNodeMap)
pool = Pool()
pool.insert(1)
pool.insert(2)
pool.insert(3)
pool.insert(4)
pool.delete(3)
print(pool.getRandom())
print(randXPowerK(3))
print(pathsToNums([9,1,4,9,0,4,8,9,0,1]))
print(unformedSum3([3,8,1,2]))
print(getString(['A', 'B', 'C'], 72))
print(getNum(['A', 'B', 'C'], "BABC"))
print(oneNums2(11))
printRandM([1,2,3,4,5,6,7,8,9], 5)
print()
print(isPlindrome(0))
print(getMin([8,8,3,4,4,5,5,6,8,8,8,8,8,8,8,8,8,8,8,8,8]))
print(isContains([4,5,6,7,1,2,3], 7))
print(getNumEngExp(-987654))
print(getNumChineseExp(-3452017))
print(candy1([1,4,5,9,3,2]))
print(candy2([1,4,5,9,3,2]))
messageBox = MessageBox()
messageBox.receive(2)
messageBox.receive(1)
messageBox.receive(4)
messageBox.receive(5)
messageBox.receive(7)
messageBox.receive(3)
messageBox.receive(9)
messageBox.receive(8)
messageBox.receive(6)
print()
medianHolder = MedianHolder()
medianHolder.addNum(6)
medianHolder.addNum(1)
medianHolder.addNum(3)
medianHolder.addNum(0)
medianHolder.addNum(9)
medianHolder.addNum(8)
medianHolder.addNum(7)
medianHolder.addNum(2)
print(medianHolder.getMedian())
medianHolder.addNum(10)
print(medianHolder.getMedian())
medianHolder.addNum(11)
print(medianHolder.getMedian())
print(getUpMedian([0,1,2], [3,4,5]))
print(findKthNum([1,2,3], [3,4,5,6], 4))
print(getTopKSum([1,2,3,4,5], [3,5,7,9,11], 4))
printTopKAndRank(['1', '1', '2', '3'], 2)
topKRecord = TopKRecord(2)
topKRecord.add("A")
topKRecord.printTopK()
topKRecord.add("B")
topKRecord.add("B")
topKRecord.printTopK()
topKRecord.add("C")
topKRecord.add("C")
topKRecord.printTopK()
print(maxLcpsLength("abc1234321ab"))
print(shortestEnd("1234567abcde"))
print(getIndexOf("acbcd", "d"))
print(chessQuestion1(10, 1))
print(chessQuestion2(10, 1))
print(chessQuestion3(10, 1))
print(chessQuestion4(10, 1))
print(painterQuestion1([1,4,1,1,1,1,1,3], 3))
print(painterQuestion2([1,4,1,1,1,1,1,3], 3))
print(painterQuestion3([1,4,1,1,1,1,1,3], 3))
print(siteSelectionQuestion1([1,2,3,4,5,6,7,8,9,1000], 2))
print(siteSelectionQuestion2([1,2,3,4,5,6,7,8,9,1000], 2))
