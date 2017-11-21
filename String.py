import sys


#判断两个字符串是否为变形词
#使用数组进行判断
def isDeformation1(str1, str2):
    if str1 ==  None or str2 == None or len(str1) != len(str2):
        return False
    array = [0 for i in range(256)]
    for i in range(len(str1)):
        array[ord(str1[i])] += 1
    for i in range(len(str2)):
        array[ord(str2[i])] -= 1
        if array[ord(str2[i])] < 0:
            return False
    return True


#使用哈希表进行判断
def isDeformation2(str1, str2):
    if str1 == None or str2 == None or len(str1) != len(str2):
        return False
    map = {}
    for i in range(len(str1)):
        if str1[i] not in map:
            map[str1[i]] = 1
        else:
            map[str1[i]] = map[str1[i]] + 1
    for i in range(len(str2)):
        if str2[i] not in map:
            return False
        else:
            map[str2[i]] = map[str2[i]] - 1
            if map[str2[i]] < 0:
                return False
    return True





#字符串中数字子串的求和
def numSum(str1):
    if str1 == None or len(str1) == 0:
        return 0
    res = 0
    num = 0
    cur = 0
    posi = True
    for i in range(len(str1)):
        cur = ord(str1[i]) - ord('0')
        if cur < 0 or cur > 9:
            res += num
            num = 0
            if str1[i] == '-':
                if i - 1 >= 0 and str1[i-1] == '-':
                    posi = not posi
                else:
                    posi = False
            else:
                pose = True
        else:
            num = num * 10 + cur if posi else -cur
    res += num
    return res





#去掉字符串中连续出现k个0的子串
def removeKZeros(str1, k):
    if str1 == None or k < 1:
        return str1
    count = 0
    start = -1
    chas = list(str1)
    for i in range(len(chas)):
        if chas[i] == '0':
            count += 1
            start = i if start == -1 else start
        else:
            if count == k:
                while count > 0:
                    chas[start] = ""
                    start += 1
                    count -= 1
            count = 0
            start = -1
    if count == k:
        while count > 0:
            chas[start] = ""
            start += 1
            count -= 1
    return ''.join(chas)





#判断两个字符串是否互为旋转词
def isRotation(str1, str2):
    def KMP(str1, str2):
        if str1 == None or str2 == None or len(str2) < 1 or len(str1) < len(str2):
            return False
        next = getNextArray(str2)
        si = 0 
        mi = 0
        while si < len(str1) and mi < len(str2):
            if str1[si] == str2[mi]:
                si += 1
                mi += 1
            elif next[mi] == -1:
                si += 1
            else:
                mi = next[mi]
        return True if mi == len(str2) else False

    def getNextArray(str1):
        if len(str1) == 1:
            return [-1]
        next = [0 for i in range(len(str1))]
        next[0] = -1
        next[1] = 0
        pos = 2
        cn = 0
        while pos < len(str1):
            if str1[pos-1] == str1[cn]:
                next[pos] = cn + 1
                pos += 1
                cn += 1
            elif cn > 0:
                cn = next[cn]
            else:
                next[pos] = 0
                pos += 1
        return next


    if str1 == None or str2 == None or len(str1) != len(str2):
        return False
    str3 = str1 * 2
    return KMP(str3, str2)





#将整数字符串转成整数型
def convert(str1):
    def isValid(str1):
        if str1[0] != "-" and (ord(str1[0]) < ord('0') or ord(str1[0]) > ord('9')):
            return False
        if str1[0] == '-' and (len(str1) == 1 or str1[1] == '0'):
            return False
        if len(str1) > 1 and str1[1] == '0':
            return False
        for i in range(1, len(str1)):
            if ord(str1[i]) < ord('0') or ord(str1[i]) > ord('9'):
                return False
        return True

    if str1 == None or str1 == "":
        return 0
    if not isValid(str1):
        return 0
    posi = False if str1[0] == '-' else True
    minq = (-1 << 31) / 10
    minr = (-1 << 31) % 10
    res = 0
    cur = 0
    for i in range(0 if posi else 1, len(str1)):
        cur = ord('0') - ord(str1[i])
        if res < minq or (res == minq and cur < minr):
            return 0
        res = res * 10 + cur
    if posi and res == (-1 << 31):
        return 0
    return -res if posi else res





#替换字符串中连续出现的指定字符串
def replace(str1, fro, to):
    if str1 == None or fro == None or to == None  or str1 == '' or fro == '':
        return str1
    chas = list(str1)
    match = 0
    for i in range(len(str1)):
        if chas[i] == fro[match]:
            match += 1
            if match == len(fro):
                index = i
                while match > 0:
                    chas[index] = ''
                    index -= 1
                    match -= 1
        else:
            match = 0
            if chas[i] == fro[0]:   #如果相等，从当前字符重新匹配
                match += 1
    cur = ''
    res = ''
    for i in range(len(str1)):
        if chas[i] != '':
            cur = cur + chas[i]
        else:
            if i == 0 or chas[i-1] != '':
                res = res + cur + to
                cur = ''
    if cur != '':
        res += cur
    return res





#字符串的统计字符串
def getCountString(str1):
    if str1 == None or str1 == '':
        return ''
    res = str1[0]
    num = 1
    for i in range(1, len(str1)):
        if str1[i] == str1[i-1]:
            num += 1
        else:
            res = res + '_' + str(num) + '_' + str1[i]
            num = 1
    return res + '_' + str(num)

def getCharAt(str1, index):
    if str1 == None or str1 == '' or index < 0:
        return ''
    posi = True
    cur = ''
    num = 0
    sum = 0
    for i in range(len(str1)):
        if str1[i] == '_':
            posi = not posi
        elif posi:
            sum += num
            if sum > index:
                return cur
            cur = str1[i]
            num = 0
        else:
            num = num * 10 +int(str1[i])
    return cur if sum+num > index else ''





#判断字符数组中是否所有的字符只出现了一次
#时间复杂度O(N)的算法
def isUnique1(arr):
    if arr == None or len(arr) == 0:
        return True
    map = {}
    for i in range(len(arr)):
        if arr[i] in map:
            return False
        map[arr[i]] = None
    return True

#额外空间复杂度O(1)的算法
def isUnique2(arr):
    def heapSort(arr):
        n = len(arr)
        for i in range((n-2)//2, -1, -1):
            precDown(arr, i, n)
        for i in range(n-1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            precDown(arr, 0, i)
        return arr

    def precDown(arr, i, n):
        child = 2 * i + 1
        tmp = arr[i]
        while child < n:
            if child < n-1 and arr[child] < arr[child+1]:
                child += 1
            if tmp < arr[child]:
                arr[i] = arr[child]
                i = child
            else:
                break
            child = 2 * i + 1
        arr[i] = tmp



    if arr == None or len(arr) == 0:
        return False
    print(heapSort([9,8,7,6,5,4,3,12,45,67]))
    heapSort(arr)
    for i in range(1, len(arr)):
        if arr[i] == arr[i-1]:
            return False
    return True





#在有序但含有空的数组中查找字符串
def getIndex(strs, str1):
    if strs == None or len(strs) == 0 or str1 == None:
        return -1
    left = 0
    right = len(strs) - 1
    res = -1
    while left <= right:
        mid = (right + left) // 2
        if strs[mid] == str1:
            res = mid
            right = mid - 1
        elif strs[mid] != None:
            if strs[mid] < str1:
                left = mid + 1
            else:
                right = mid - 1
        else:
            i = mid
            while i >= left:
                if strs[i] != None:
                    break
                i -= 1
            if strs[i] < str1:
                left = mid + 1
            else:
                res = i if strs[i] == str1 else res
                right = i - 1
    return res





#字符串的调整与替换
#原问题
def adjustAndReplace(chas):
    if chas == None or len(chas) == 0:
        return chas
    length = 0
    num = 0
    for i in range(len(chas)):
        if chas[i] != '':
            length += 1
            if chas[i] == ' ':
                num += 1
        else:
            break
    newlen = length + 2 * num
    for i in range(length-1, -1, -1):
        if chas[i] != ' ':
            chas[newlen-1] = chas[i]
            newlen -= 1
        else:
            chas[newlen-1] = '0'
            chas[newlen-2] = '2'
            chas[newlen-3] = '%'
            newlen -= 3
    return ''.join(chas)

#补充问题
def modify(chas):
    if chas == None or len(chas) == 0:
        return chas
    n = len(chas)
    for i in range(len(chas)-1, -1, -1):
        if chas[i] != '*':
            chas[n-1] = chas[i]
            n -= 1
    for i in range(n-1, -1, -1):
        chas[i] = '*'
    return chas





#翻转字符串
#原问题
def rotateWord(chas):
    def reverse(chas, start, end):
        while start < end:
            chas[start], chas[end] = chas[end], chas[start]
            start += 1
            end -= 1


    if chas == None or len(chas) == 0:
        return 
    reverse(chas, 0, len(chas)-1)
    left = -1
    right = -1
    for i in range(len(chas)):
        if chas[i] != ' ':
            left = i if i == 0 or chas[i-1] == ' ' else left
            right = i if i ==  len(chas)-1 or chas[i+1] == ' ' else right
        if left != -1 and right != -1:
            reverse(chas, left, right)
            left = -1
            right = -1
    return chas

#补充问题
def rotate1(chas, size):
    def exchange(chas, start, end, size):
        i = end - size + 1
        while size > 0:
            chas[start], chas[i] = chas[i], chas[start]
            start += 1
            i += 1
            size -= 1


    if chas == None or len(chas) == 0 or size < 0:
        return 
    start = 0
    end = len(chas) - 1
    lpart = size
    rpart = len(chas) - size
    s = min(lpart, rpart)
    d = lpart - rpart
    while True:
        exchange(chas, start, end, s)
        if d == 0:
            break
        elif d < 0:
            rpart = -d
            end -= s
        else:
            lpart = d
            start += s
        s = min(lpart, rpart)
        d = lpart - rpart
    return chas


def rotate2(chas, size):
    def reverse(chas, start, end):
        while start < end:
            chas[start], chas[end] = chas[end], chas[start]
            start += 1
            end -= 1


    if chas == None or len(chas) == 0 or size < 0:
        return
    reverse(chas, 0, size-1)
    reverse(chas, size, len(chas)-1)
    reverse(chas, 0, len(chas)-1)
    return chas





#数组中两个字符串的最小距离
def minDistance(strs, str1, str2):
    if strs == None or str1 == None or str2 == None:
        return -1
    if str1 == str2:
        return 0
    last1 = -1
    last2 = -1
    minDistance = sys.maxsize
    for i in range(len(strs)):
        if strs[i] == str1:
            if last2 != -1:
                dist = i - last2
                minDistance = min(minDistance, dist)
            last1 = i
        if strs[i] == str2:
            if last1 != -1:
                dist = i - last1
                minDistance = min(minDistance, dist)
            last2 = i
    return minDistance if minDistance != sys.maxsize else -1





#添加最少字符使字符串整体都是回文字符串
#原问题
def getPalindrome(str1):
    def getdp(str1):
        dp = [[0 for i in range(len(str1))] for j in range(len(str1))]
        for j in range(1, len(str1)):
            dp[j-1][j] = 0 if str1[j-1] == str1[j] else 1
            for i in range(j-2, -1, -1):
                if str1[i] == str1[j]:
                    dp[i][j] = dp[i+1][j-1]
                else:
                    dp[i][j] = min(dp[i+1][j], dp[i][j-1]) + 1
        return dp


    if str1 == None or len(str1) < 2:
        return str1
    dp = getdp(str1)
    res = [0 for i in range(len(str1)+dp[0][len(str1)-1])]
    i = 0
    j = len(str1) - 1
    resl = 0
    resr = len(res) - 1
    while i <= j:
        if str1[i] == str1[j]:
            res[resl] = str1[i]
            res[resr] = str1[j]
            i += 1
            j -= 1
        elif dp[i+1][j] < dp[i][j-1]:
            res[resl] = str1[i]
            res[resr] = str1[i]
            i += 1
        else:
            res[resl] = str1[j]
            res[resr] = str1[j]
            j -= 1
        resl += 1
        resr -= 1
    return ''.join(res)

#补充问题
def getPalindrome2(str1, strlps):
    if str1 == None or len(str1) == 0 or strlps == None or len(strlps) == 0:
        return 
    res = [0 for i in range(2*len(str1)-len(strlps))]
    lstr = 0
    rstr = len(str1)-1
    llps = 0
    rlps = len(strlps)-1
    lres = 0
    rres = len(res)-1
    while llps <= rlps:
        temp1 = lstr
        temp2 = rstr
        while str1[lstr] != strlps[llps]:
            lstr += 1
        while str1[rstr] != strlps[rlps]:
            rstr -= 1
        for i in range(temp1, lstr):
            res[lres] = str1[i]
            res[rres] = str1[i]
            lres += 1
            rres -= 1
        for i in range(temp2, rstr, -1):
            res[lres] = str1[i]
            res[rres] = str1[i]
            lres += 1
            rres -= 1
        res[lres] = str1[lstr]
        res[rres] = str1[rstr]
        lstr += 1
        rstr -= 1
        lres += 1
        rres -= 1
        llps += 1
        rlps -= 1
    return ''.join(res)





#括号字符串的有效性和最长有效长度
#原题目
def isValid(str1):
    if str1 == None or len(str1) == 0:
        return False
    status = 0
    for i in range(len(str1)):
        if str1[i] != '(' and str1[i] != ')':
            return False
        elif str1[i] == '(':
            status += 1
        else:
            status -= 1
            if status < 0:
                return False
    return status == 0

#补充题目
def maxValidLength(str1):
    if str1 == None or len(str1) == 0:
        return 0
    dp = [0 for i in range(len(str1))]
    res = 0
    for i in range(1, len(str1)):
        if str1[i] == ')':
            pre = i - dp[i-1] - 1
            if pre >= 0 and str1[pre] == '(':
                dp[i] = dp[i-1] + 2 + (dp[pre-1] if pre > 0 else 0)
            res = max(res, dp[i])
    return res





#公式字符串求值
def getValue(exp):
    def value(exp, i):
        deque = []
        pre = 0      
        while i < len(exp) and exp[i] != ')':
            if ord(exp[i]) >= ord('0') and ord(exp[i]) <= ord('9'):
                pre = pre * 10 + int(exp[i])
                i += 1
            elif exp[i] != '(':
                addNum(deque, pre)
                deque.append(exp[i])
                i += 1
                pre = 0
            else:
                bra = value(exp, i + 1)
                pre = bra[0]
                i = bra[1] + 1
        addNum(deque, pre)
        return [getNum(deque), i]

    #计算乘法和除法
    def addNum(deque, pre):
        if deque:
            top = deque.pop()
            if top == '+' or top == '-':
                deque.append(top)
            else:
                cur = int(deque.pop())
                pre = cur * pre if top == '*' else cur / pre
        deque.append(pre)

    #计算加法和减法
    def getNum(deque):
        res = 0
        add = True
        while deque:
            cur = deque.pop(0)
            if cur == '+':
                add = True
            elif cur == '-':
                add = False
            else:
                res += int(cur) if add else -int(cur)
        return res


    return value(exp, 0)[0]





#0左边必定有1的二进制字符串数量
#递归
def getNum1(n):
    if n < 1:
        return 0
    if n == 1 or n == 2:
        return n
    return getNum1(n-1) + getNum1(n-2)

#动态规划
def getNum2(n):
    if n < 1:
        return 0
    if n == 1 or n == 2:
        return n
    pre = 1
    cur = 2
    for i in range(3, n+1):
        pre, cur = cur, pre+cur
    return cur

#矩阵乘法
def getNum3(n):
    def matrixPower(m, p):
        res = [[0 if i != j else 1 for i in range(len(m[0]))] for j in range(len(m))]
        tmp = m
        while p > 0:
            if p & 1 != 0:
                res = muliMatrix(res, tmp)
            tmp = muliMatrix(tmp, tmp)
            p >>= 1
        return res

    def muliMatrix(m1, m2):
        res = [[0 for i in range(len(m2[0]))] for j in range(len(m1))]
        for i in range(len(m1)):
            for j in range(len(m2[0])):
                for k in range(len(m1[0])):
                    res[i][j] += m1[i][k] * m2[k][j]
        return res

    if n < 1:
        return 0
    if n == 1 or n == 2:
        return n
    base = [[1,1], [1,0]]
    res = matrixPower(base, n-2)
    return 2 * res[0][0] + res[1][0]





#拼接所有字符串产生字典顺序最小的大字符串
def lowestString(chas):
    if chas == None or len(chas) == 0:
        return ""
    from functools import cmp_to_key
    chas = sorted(chas, key=cmp_to_key(lambda x,y: 1 if x+y > y+x else -1))
    return ''.join(chas)





#找到字符串的最长无重复字符子串
def maxUniqueStr(str1):
    if str1 == None or len(str1) == 0:
        return ""
    map = [-1 for i in range(256)]
    pre = -1
    length = 0
    for i in range(len(str1)):
        pre = max(pre, map[ord(str1[i])])
        length = max(length, i-pre)
        map[ord(str1[i])] = i
    return length

def maxUniqueStr2(str1):
    if str1 == None or len(str1) == 0:
        return ""
    map = {}
    pre = -1
    length = 0
    for i in range(len(str1)):
        if str1[i] in map:
            pre = max(pre, map[str1[i]])
        length = max(length, i-pre)
        map[str1[i]] = i
    return length





#找到被指的新类型字符
def pointNewChar(str1, k):
    if str1 == None or len(str1) == 0 or k < 0 or k >= len(str1):
        return ""
    uNum = 0
    for i in range(k-1, -1, -1):
        if str1[i].islower():
            break
        uNum += 1

    if uNum & 1 == 1:
        return str1[k-1 : k+1]
    elif str1[k].isupper():
        return str1[k : k+2]
    else:
        return str1[k]





#最小包含子串的长度
def minLength(str1, str2):
    if str1 == None or str2 == None or len(str1) < len(str2):
        return 0
    map = [0 for i in range(256)]
    for i in range(len(str2)):
        map[ord(str2[i])] += 1
    left = 0
    right = 0
    match = len(str2)
    minlength = sys.maxsize
    while right < len(str1):
        map[ord(str1[right])] -= 1
        if map[ord(str1[right])] >= 0:
            match -= 1
        if match == 0:
            while map[ord(str1[left])] < 0:
                map[ord(str1[left])] += 1
            minlength = min(minlength, right - left + 1)
            match += 1
            map[ord(str1[left])] += 1
            left += 1
        right += 1
    return minlength if minlength != sys.maxsize else 0





#回文最小分割数
def minCut(str1):
    if str1 == None or str1 == "":
        return 0
    N = len(str1)
    dp = [0 for i in range(N)]
    p = [[False for i in range(N)] for j in range(N)]
    for i in range(N-1, -1, -1):
        dp[i] = sys.maxsize
        for j in range(i, N):
            if str1[i] == str1[j] and (j-i < 2 or p[i+1][j-1]):
                p[i][j] = True
                dp[i] = min(dp[i], 0 if j+1 == N else dp[j+1] + 1)
    return dp[0]

def minCut2(str1):
    if str1 == None or str1 == "":
        return 0
    N = len(str1)
    p = [[False for i in range(N)] for j in range(N)]
    dp = [0 for i in range(N)]
    for i in range(N):
        dp[i] = sys.maxsize
        for j in range(i, -1, -1):
            if str1[j] == str1[i] and (i-j < 2 or p[j+1][i-1]):
                p[j][i] = True
                dp[i] = min(dp[i], 0 if j-1 == -1 else dp[j-1] + 1)
    return dp[-1]

    


print(isDeformation1("132", "123"))
print(numSum("A-1B--2C--D6E"))
print(removeKZeros("A00B", 2))
print(isRotation("2ab1", "ab12"))
print(convert("-1234518"))
print(replace("123abcabc", "abc", "X"))
print(getCountString('aaabbadddffc'))
print(getCharAt('a_3_b_3_c_4', 5))
print(isUnique2(['1','a','4','1']))
print(getIndex([None, 'a', None, 'b', None, 'c'], 'b'))
print(adjustAndReplace(['a',' ','b', '', '', '', '']))
print(modify(['a', 'b', 'c', '*', '*', 'd', '*']))
print(rotateWord(list("I love jijianli")))
print(rotate2(list("1234567abc"), 7))
print(minDistance("1333231",'1','3'))
print(getPalindrome2("A1B21C", "121"))
print(isValid("()()()()"))
print(maxValidLength("(()())"))
print(getValue("48*((70-65)-43)+8*1"))
print(getNum1(3))
print(lowestString(["de", "abc"]))
print(maxUniqueStr2("aabcb"))
print(pointNewChar("aaABCDEcBCg", 10))
print(minLength("abcde", "ad"))
print(minCut2("ADA"))

