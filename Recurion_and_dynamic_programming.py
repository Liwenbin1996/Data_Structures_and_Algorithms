import sys
#斐波那契系列问题的递归与动态规划

#fibonacci问题
def fibonacci1(n):
    if n < 1:
        return 0
    if n == 1 or n == 2:
        return 1
    return fibonacci1(n-1) + fibonacci1(n-2)

def fibonacci2(n):
    if n < 1:
        return 0
    if n == 1 or n == 2:
        return 1
    pre = 1
    cur = 1
    for i in range(3, n+1):
        pre, cur = cur, pre+cur
    return cur

def fibonacciUseMatrix(n):
    def matrixPower(m, p):
        res = [[0 if i != j else 1 for i in range(len(m[0]))] for j in range(len(m))]  #单位矩阵
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
        return 1
    base = [[1,1],[1,0]]
    res = matrixPower(base, n-2)
    return res[0][0] + res[1][0]


#母牛数量问题
def fibonacci3(n):
    if n < 1 :
        return 0
    if n == 1 or n == 2 or n == 3:
        return n
    return fibonacci3(n-1) + fibonacci3(n-3)

def fibonacci4(n):
    if n < 1:
        return 0
    if n == 1 or n == 2 or n == 3:
        return 3
    prepre = 1
    pre = 2
    cur = 3
    for i in range(4, n+1):
        tmp = cur
        cur = prepre + cur
        prepre = pre
        pre = tmp
    return cur

def fibonacciUseMatrix2(n):
    def matrixPower(m, p):
        res = [[1 if i == j else 0 for i in range(len(m[0]))] for j in range(len(m))]
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
    if n == 1 or n ==2 or n == 3:
        return n
    base = [[1,1,0], [0,0,1], [1,0,0]]
    res = matrixPower(base, n-3)
    return 3 * res[0][0] + 2 * res[1][0] + res[2][0]





#矩阵的最小路径和
def minPathSum1(m):
    if m == None or len(m) == 0 or m[0] == None or len(m[0]) == 0:
        return 0
    dp = [[0 for i in range(len(m[0]))] for j in range(len(m))]
    dp[0][0] = m[0][0]
    for i in range(1, len(m[0])):
        dp[0][i] = dp[0][i-1] + m[0][i]
    for j in range(1, len(m)):
        dp[j][0] = dp[j-1][0] + m[j][0]
    for i in range(1, len(m)):
        for j in range(1, len(m[0])):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + m[i][j]
    return dp[len(m)-1][len(m[0])-1]

def minPathSum2(m):
    if m == None or len(m) == 0 or m[0] == None or len(m[0]) == 0:
        return 0
    more = max(len(m), len(m[0]))
    less = min(len(m), len(m[0]))
    rowmore = True if more == len(m) else False
    dp = [0 for i in range(less)]
    dp[0] = m[0][0]
    for i in range(1, less):
        dp[i] = dp[i-1] + m[0][i] if rowmore else m[i][0]
    for i in range(1, more):
        dp[0] = dp[0] + m[i][0] if rowmore else m[0][i]
        for j in range(1, less):
            dp[j] = min(dp[j-1], dp[j]) + m[i][j] if rowmore else m[j][i]
    return dp[less-1]





#换钱的最少货币数
def minCoins1(arr, aim):
    if arr == None or len(arr) == 0 or aim < 0:
        return -1
    row = len(arr)
    dp = [[sys.maxsize for i in range(aim+1)] for j in range(row)]
    for i in range(row):
        dp[i][0] = 0
    for j in range(1, aim+1):
        if j % arr[0] == 0:
            dp[0][j] = j // arr[0]
    for i in range(1, row):
        for j in range(1, aim+1):
            left = sys.maxsize
            if j - arr[i] >= 0 and dp[i][j-arr[i]] != sys.maxsize:
                left = dp[i][j-arr[i]] + 1
            dp[i][j] = min(left, dp[i-1][j])
    return dp[row-1][aim] if dp[row-1][aim] != sys.maxsize else -1

def minCoins2(arr, aim):
    if arr == None or len(arr) == 0 or aim < 0:
        return -1
    row = len(arr)
    dp = [sys.maxsize for i in range(aim+1)]
    dp[0] = 0
    for i in range(1, aim+1):
        if i % arr[0] == 0:
            dp[i] = i // arr[0]
    for i in range(1, row):
        dp[0] = 0
        for j in range(1, aim+1):
            left = sys.maxsize
            if j - arr[i] >= 0 and dp[j-arr[i]] != sys.maxsize:
                left = dp[j-arr[i]] + 1
            dp[j] = min(left, dp[j])
    return dp[aim] if dp[aim] != sys.maxsize else -1





#换钱的最少货币数(每种货币只有一张)
def minCoins3(arr, aim):
    if arr == None or len(arr) == 0 or aim < 0:
        return -1
    row = len(arr)
    dp = [[sys.maxsize for i in range(aim+1)] for j in range(len(arr))]
    for i in range(len(arr)):
        dp[i][0] = 0
    if arr[0] <= aim:
        dp[0][arr[0]] = 1
    for i in range(1, len(arr)):
        for j in range(1, aim+1):
            leftUp = sys.maxsize
            if j-arr[i] >= 0 and dp[i-1][j-arr[i]] != sys.maxsize:
                leftUp = dp[i-1][j-arr[i]] + 1
            dp[i][j] = min(leftUp, dp[i-1][j])
    return dp[row-1][aim] if dp[row-1][aim] != sys.maxsize else -1

def minCoins4(arr, aim):
    if arr == None or len(arr) == 0 or aim < 0:
        return -1
    row = len(arr)
    dp = [sys.maxsize for i in range(aim+1)]
    dp[0] = 0
    if arr[0] <= aim:
        dp[arr[0]] = 1
    for i in range(1, row):
        for j in range(1, aim+1):
            leftUp = sys.maxsize
            if j-arr[i] >= 0 and dp[j-arr[i]] != sys.maxsize:
                leftUp = dp[j-arr[i]] + 1
            dp[j] = min(leftUp, dp[j])
    return dp[aim] if dp[aim] != sys.maxsize else -1





#换钱的方法数

#暴力递归方法
def coins1(arr, aim):
    def process1(arr, index, aim):
        if index == len(arr):
            return 1 if aim == 0 else 0
        else:
            res = 0
            for i in range(0, aim//arr[index]+1):
                res += process1(arr, index+1, aim-arr[index]*i)
        return res


    if arr == None or len(arr) == 0 or aim < 0:
        return 0
    return process1(arr, 0, aim)

#记忆搜索方法
def coins2(arr, aim):
    def process2(arr, index, aim, records):
        if index == len(arr):
            return 1 if aim == 0 else 0
        else:
            res = 0
            for i in range(0, aim//arr[index]+1):
                mapValue = records[index+1][aim-arr[index]*i]
                if mapValue != 0:
                    res += mapValue if mapValue != -1 else 0
                else:
                    res += process2(arr, index+1, aim-arr[index]*i, records)
        records[index][aim] = -1 if res == 0 else res
        return res


    if arr == None or len(arr) == 0 or aim < 0:
        return 0
    records = [[0 for i in range(aim+1)] for j in range(len(arr)+1)]
    return process2(arr, 0, aim, records)

#动态规划方法
def coins3(arr, aim):
    if arr == None or len(arr) == 0 or aim < 0:
        return 0
    row = len(arr)
    dp = [[0 for i in range(aim+1)]for j in range(row)]
    for i in range(row):
        dp[i][0] = 1
    for j in range(1, aim//arr[0]+1):
        dp[0][arr[0]*j] = 1
    for i in range(1, row):
        for j in range(1, aim+1):
            num = 0
            for k in range(j//arr[i]+1):
                num += dp[i-1][j-arr[i]*k]
            dp[i][j] = num
    return dp[row-1][aim]

#动态规划升级版
def coins4(arr, aim):
    if arr == None or len(arr) == 0 or aim < 0:
        return 0
    row = len(arr)
    dp = [[0 for i in range(aim+1)] for j in range(row)]
    for i in range(row):
        dp[i][0] = 1
    for j in range(1, aim//arr[0]+1):
        dp[0][arr[0]*j] = 1
    for i in range(1,row):
        for j in range(1, aim+1):
            dp[i][j] = dp[i-1][j]
            dp[i][j] += dp[i][j-arr[i]] if j-arr[i] >= 0 else 0
    return dp[row-1][aim]

#动态规划升级版+空间压缩
def coins5(arr, aim):
    if arr == None or len(arr) == 0 or aim < 0:
        return 0
    dp = [0 for i in range(aim+1)]
    for i in range(aim//arr[0]+1):
        dp[arr[0]*i] = 1
    for i in range(1, len(arr)):
        for j in range(1, aim+1):
            dp[j] += dp[j-arr[i]] if j-arr[i] >= 0 else 0
    return dp[aim]





#最长递增子序列
def getMaxSubList1(arr):
    def getdp(arr):
        dp = [1 for i in range(len(arr))]
        for i in range(len(arr)):
            for j in range(i):
                if arr[i] > arr[j]:
                    dp[i] = max(dp[i], dp[j]+1)
        return dp

    def generateLIS(arr, dp):
        maxlen = 0
        index = 0
        for i in range(len(dp)):
            if dp[i] > maxlen:
                maxlen = dp[i]
                index = i
        lis = [0 for i in range(maxlen)]
        lis[maxlen-1] = arr[index]
        maxlen -= 1
        for i in range(index, -1, -1):
            if arr[i] < arr[index] and dp[i]+1 == dp[index]:
                lis[maxlen-1] = arr[i]
                maxlen -= 1
                index = i
        return lis


    if arr == None or len(arr) == 0:
        return None
    dp = getdp(arr)
    return generateLIS(arr, dp)

def getMaxSubList2(arr):
    def getdp2(arr):
        dp = [0 for i in range(len(arr))]
        ends = [0 for i in range(len(arr))]
        right = 0
        dp[0] = 1
        ends[0] = arr[0]
        for i in range(1, len(arr)):
            l = 0
            r = right
            while l <= r:
                m = (l + r) // 2
                if arr[i] > ends[m]:
                    l = m + 1
                else:
                    r = m - 1
            right = max(right, l)
            dp[i] = l + 1
            ends[l] = arr[i]
        return dp

    def generateLIS(arr, dp):
        maxlen = 0
        index = 0
        for i in range(len(dp)):
            if dp[i] > maxlen:
                maxlen = dp[i]
                index = i
        lis = [0 for i in range(maxlen)]
        lis[maxlen-1] = arr[index]
        maxlen -= 1
        for i in range(index, -1, -1):
            if arr[i] < arr[index] and dp[i]+1 == dp[index]:
                lis[maxlen-1] = arr[i]
                maxlen -= 1
                index = i
        return lis


    if arr == None or len(arr) == 0:
        return None
    dp = getdp2(arr)
    return generateLIS(arr, dp)





#汉诺塔问题
def hanoi(n):
    def func(n, left, mid, right):
        if n == 1:
            print("move from " + left + " to " + right)
        else:
            func(n-1, left, right, mid)
            func(1, left, mid, right)
            func(n-1, mid, left, right)


    if n < 1:
        return
    return func(n, "left", "mid", "right")




#最长公共子序列问题
def maxCommonSubSerial(str1, str2):
    def getdp(str1, str2):
        dp = [[0 for i in range(len(str2))] for j in range(len(str1))]
        dp[0][0] = 1 if str1[0] == str2[0] else 0
        for i in range(1, len(str2)):
            dp[0][i] = max(dp[0][i-1], 1 if str1[0] == str2[i] else 0)
        for i in range(1, len(str1)):
            dp[i][0] = max(dp[i-1][0], 1 if str1[i] == str2[0] else 0)
        for i in range(1, len(str1)):
            for j in range(1, len(str2)):
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                if str1[i] == str2[j]:
                    dp[i][j] = max(dp[i-1][j-1]+1, dp[i][j])
        return dp


    if str1 == None or str2 == None or str1 == "" or str2 == "":
        return ""
    dp = getdp(str1, str2)
    m = len(str2) - 1
    n = len(str1) - 1
    res = [0 for i in range(dp[n][m])]
    index = dp[n][m] - 1
    while index >= 0:
        if n > 0 and dp[n][m] == dp[n-1][m]:
            n -= 1
        elif m > 0 and dp[n][m] == dp[n][m-1]:
            m -= 1
        else:
            res[index] = str1[n]
            index -= 1
            m -= 1
            n -= 1
    return ''.join(res)




#最长公共子串问题
#经典动态规划方法。时间复杂度O(M*N)，空间复杂度O(M*N)
def maxCommonSubStr(str1, str2):
    def getdp(str1, str2):
        dp = [[0 for i in range(len(str2))] for j in range(len(str1))]
        for i in range(len(str2)):
            if str2[i] == str1[0]:
                dp[0][i] = 1
        for i in range(len(str1)):
            if str1[i] == str2[0]:
                dp[i][0] = 1
        for i in range(1, len(str1)):
            for j in range(1, len(str2)):
                if str1[i] == str2[j]:
                    dp[i][j] = dp[i-1][j-1] + 1
        return dp


    if str1 == None or str2 == None or str1 == '' or str2 == '':
        return ""
    dp = getdp(str1, str2)
    length = 0
    index = 0
    for i in range(len(str1)):
        for j in range(len(str2)):
            if dp[i][j] > length:
                length = dp[i][j]
                index = i
    return str1[index-length+1 : index+1]

#动态规划升级版。时间复杂度O(M*N)，空间复杂度O(1)
def maxCommonSubStr2(str1, str2):
    if str1 == None or str2 == None or str1 == "" or str2 == "":
        return ""
    maxlen = 0
    index = 0
    row = 0
    col = len(str2) - 1
    while row < len(str1):
        i = row 
        j = col
        length = 0
        while i < len(str1) and j < len(str2):
            if str1[i] == str2[j]:
                length += 1
            else:
                length = 0
            if length > maxlen:
                maxlen = length
                index = i
            i += 1
            j += 1
        if col > 0:
            col -= 1
        else:
            row += 1
    return str1[index-maxlen+1 : index+1]





#最小编辑代价
#经典动态规划方法。时间复杂度O(M*N),空间复杂度O(M*N)
def minCoin(str1, str2, ic, dc, rc):
    if str1 == None or str2 == None:
        return 0
    row = len(str1) + 1
    col = len(str2) + 1
    dp = [[0 for i in range(col)] for j in range(row)]
    for i in range(row):
        dp[i][0] = i * dc
    for j in range(col):
        dp[0][j] = j * ic
    for i in range(1, row):
        for j in range(1, col):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = dp[i-1][j-1] + rc
            dp[i][j] = min(dp[i][j], dp[i][j-1] + ic)
            dp[i][j] = min(dp[i][j], dp[i-1][j] + dc)
    return dp[-1][-1]

#动态规划+空间压缩。时间复杂度O(M*N)，空间复杂度O(min{M,N})
def minCoin2(str1, str2, ic, dc, rc):
    if str1 == None or str2 == None:
        return 0
    longs = str1 if len(str1) >= len(str2) else str2
    shorts = str1 if len(str1) < len(str2) else str2
    if longs == str2:
        ic, dc = dc, ic
    dp = [0 for i in range(len(shorts)+1)]
    for i in range(len(dp)):
        dp[i] = ic * i
    for i in range(1, len(longs)+1):
        leftUp = dp[0]
        dp[0] = i * dc
        for j in range(1, len(shorts)+1):
            up = dp[j]
            if longs[i-1] == shorts[j-1]:
                dp[j] = leftUp
            else:
                dp[j] = leftUp + rc
            dp[j] = min(dp[j], up+dc)
            dp[j] = min(dp[j], dp[j-1]+ic)
            leftUp = up
    return dp[-1]





#字符串的交错组成
#经典动态规划方法
def isCross1(str1, str2, aim):
    if str1 == None or str2 == None or aim == None or len(str1)+len(str2) != len(aim):
        return False
    dp = [[False for i in range(len(str2)+1)] for j in range(len(str1)+1)]
    dp[0][0] = True
    for i in range(1, len(str1)+1):
        if str1[i-1] != aim[i-1]:
            break
        dp[i][0] = True
    for j in range(1, len(str2)+1):
        if str2[j-1] != aim[j-1]:
            reak
        dp[0][j] = True
    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if (dp[i-1][j] == True and str1[i-1] == aim[i+j-1]) or (dp[i][j-1] == True and str2[j-1] == aim[i+j-1]):
                dp[i][j] = True
    return dp[-1][-1]

#动态规划+空间压缩
def isCross2(str1, str2, aim):
    if str1 == None or str2 == None or aim == None or len(str1)+len(str2) != len(aim):
        return False
    longs = str1 if len(str1) >= len(str2) else str2
    shorts = str1 if len(str1) < len(str2) else str2
    dp = [False for i in range(len(shorts)+1)]
    dp[0] = True
    for i in range(1, len(dp)):
        if shorts[i-1] != aim[i-1]:
            break
        dp[i] = True
    for i in range(1, len(longs)+1):
        for j in range(1, len(shorts)+1):
            if (dp[j-1] == True and shorts[j-1] == aim[i+j-1]) or (dp[i] == True and longs[i-1] == aim[i+j-1]):
                dp[j] = True
    return dp[-1]





#龙与地下城游戏问题
#经典动态规划方法
def minHP1(mat):
    if mat == None or mat[0] == None or len(mat) == 0 or len(mat[0]) == 0:
        return 1
    row = len(mat)
    col = len(mat[0])
    dp = [[0 for i in range(col)] for j in range(row)]
    dp[row-1][col-1] = max(-mat[row-1][col-1]+1, 1)
    for i in range(row-2, -1, -1):
        dp[i][col-1] = max(dp[i+1][col-1] - mat[i][col-1], 1)
    for j in range(col-2, -1, -1):
        dp[row-1][j] = max(dp[row-1][j+1] - mat[row-1][j], 1)
    for i in range(row-2, -1, -1):
        for j in range(col-2, -1, -1):
            right = max(dp[i][j+1] - mat[i][j], 1)
            down = max(dp[i+1][j] - mat[i][j], 1)
            dp[i][j] = min(right, down)
    return dp[0][0]

#动态规划＋空间压缩
def minHP2(mat):
    if mat == None or mat[0] == None or len(mat[0]) == 0 or len(mat) == 0:
        return 1
    more = len(mat) if len(mat) >= len(mat[0]) else len(mat[0])
    less = len(mat) if len(mat) < len(mat[0]) else len(mat[0])
    rowmore = True if more == len(mat) else False
    dp = [0 for i in range(less)]
    dp[-1] = max(-mat[more-1][less-1]+1, 1)
    for j in range(less-2, -1, -1):
        row = more-1 if rowmore else j
        col = j if rowmore else more-1
        dp[j] = max(dp[j+1] - mat[row][col], 1)
    for i in range(more-2, -1, -1):
        row = i if rowmore else less-1
        col = less-1 if rowmore else i
        dp[-1] = max(dp[-1] - mat[row][col], 1)
        for j in range(less-2, -1, -1):
            row = i if rowmore else j
            col = j if rowmore else i
            right = max(dp[j+1] - mat[row][col], 1)
            down = max(dp[j] - mat[row][col], 1)
            dp[j] = min(right, down)
    return dp[0]





#数字字符串转换为字母组合的种数
#暴力递归方法
def num1(str1):
    def process1(str1, i):
        if i == len(str1):
            return 1
        if str1[i] == '0':
            return 0
        res = process1(str1, i+1)
        if i+1 < len(str1) and ((int(str1[i])) * 10 + int(str1[i+1])) < 27:
            res += process1(str1, i+2)
        return res



    if str1 == None or str1 == "":
        return 0
    return process1(str1, 0)

#动态规划方法
def num2(str1):
    if str1 == None or str1 == "":
        return 0
    cur = 1 if str1[-1] != '0' else 0
    nex = 1
    for i in range(len(str1)-2, -1, -1):
        if str1[i] == '0':
            nex = cur
            cur = 0
        else:
            tmp = cur
            if int(str1[i]) * 10 + int(str1[i+1]) < 27:
                cur += nex
            nex = tmp
    return cur





#排成一条线的纸牌博弈问题
#暴力递归方法
def win1(arr):
    def f(arr, start, end):
        if start == end:
            return arr[start]
        return max(arr[start] + s(arr, start+1, end), arr[end] + s(arr, start, end-1))

    def s(arr, start, end):
        if start == end:
            return 0
        return min(f(arr, start+1, end), f(arr, start, end-1))


    if arr == None or len(arr) == 0:
        return 0
    return max(f(arr, 0, len(arr)-1), s(arr, 0, len(arr)-1))

#动态规划方法
def win2(arr):
    if arr == None or len(arr) == 0:
        return 0
    f = [[0 for i in range(len(arr))] for j in range(len(arr))]
    s = [[0 for i in range(len(arr))] for j in range(len(arr))]
    for j in range(len(arr)):
        f[j][j] = arr[j]
        s[j][j] = 0
        for i in range(j-1, -1, -1):
            f[i][j] = max(arr[i] + s[i+1][j], arr[j] + s[i][j-1])
            s[i][j] = min(f[i+1][j], f[i][j-1])
    return max(f[0][len(arr)-1], s[0][len(arr)-1])





#跳跃游戏
def jump(arr):
    if arr == None or len(arr) == 0:
        return 0
    jump = 0
    distance = 0
    next = 0
    for i in range(len(arr)):
        if distance < i:
            jump += 1
            distance = next
        next = max(i+arr[i], next)
    return jump





#数组中的最长连续序列
def longestConsecutive(arr):
    def merge(map, less, more):
        left = less - map[less] +1
        right = more + map[more] -1
        length = right - left + 1
        map[left] = length
        map[right] = length
        return length


    if arr == None or len(arr) == 0:
        return 0
    map = {}
    maxstr = 1 
    for i in range(len(arr)):
        if arr[i] not in map:
            map[arr[i]] = 1
            if arr[i]-1 in map:
                maxstr = max(maxstr, merge(map, arr[i]-1, arr[i]))
            if arr[i]+1 in map:
                maxstr = max(maxstr, merge(map, arr[i], arr[i]+1))
    return maxstr





#N皇后问题
#递归方法。
def queens(n):
    def isValid(record, i, j):
        for k in range(i):
            if record[k] == j or abs(record[k] - j) == abs(i - k):
                return False
        return True

    def process(index, record):       #对棋牌的第index行进行判断
        if index == len(record):
            return 1
        res = 0
        for j in range(len(record)):      #对棋牌的每一列进行判断
            if isValid(record, index, j):
                record[index] = j
                res += process(index+1, record)
        return res


    if n < 1:
        return 0
    record = [0 for i in range(n)]
    return process(0, record)

#使用位运算进行加速的递归方法。
def queens2(n):
    def process2(upperLim, colLim, leftDiaLim, rightDiaLim):
        if colLim == upperLim:
            return 1
        res = 0
        pos = upperLim & ~(colLim | leftDiaLim | rightDiaLim)
        while pos != 0:
            mostRightOne = pos & (~pos + 1)
            pos = pos - mostRightOne
            res += process2(upperLim, colLim | mostRightOne, (leftDiaLim | mostRightOne) << 1, (rightDiaLim | mostRightOne) >> 1)
        return res


    if n < 1 or n > 32:          #位运算的载体是int型变量，只能求解1~32皇后问题
        return 0
    upperLim = -1 if n == 32 else (1<<n) - 1
    return process2(upperLim, 0, 0, 0)



print(fibonacci1(6))
print(fibonacciUseMatrix(6))
print(fibonacci3(6))
print(fibonacciUseMatrix2(6))
test1 = [[1,3,5,9],[8,1,3,4],[5,0,6,1],[8,8,4,0]]
print(minPathSum1(test1))
print(minCoins4([5,2,3], 1))
print(coins5([5,10,25,1], 15))
print(getMaxSubList2([2,1,5,3,6,4,8,9,7]))
hanoi(2)
print(maxCommonSubSerial("1A2C3D4B56", "B1D23CA45B6A"))
print(maxCommonSubStr2("1AB2345CD", "12345EF"))
print(minCoin("abc", "adc", 5, 3, 100))
print(isCross2("AB", "12", "AB2"))
print(minHP2([[-2, -3, 3],[-5, -10, 1],[0, 30, -5]]))
print(num2('1111'))
print(win2([1, 2, 100, 4]))
print(jump([3,2,3,1,1,4]))
print(longestConsecutive([100,4,200,1,3,2]))
print(queens2(6))
