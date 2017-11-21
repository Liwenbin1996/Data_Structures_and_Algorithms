import sys
class TreeNode:
    def __init__(self,x):
        self.val = x
        self.left = None
        self.right = None

def create_tree(root):
    element = input("Enter a key: ")
    if element == '#':
        root = None
    else:
        root = TreeNode(element)
        root.left = create_tree(root.left)
        root.right = create_tree(root.right)
    return root

def createTreeByPreOrder(arr):
    key = arr.pop(0)
    if key == '#':
        return None
    root = TreeNode(key)
    root.left = createTreeByPreOrder(arr)
    root.right = createTreeByPreOrder(arr)
    return root

#直观的打印二叉树
def printTree(root):
    if not root:
        return
    print("Binary Tree: ")
    printInOrder(root, 0, 'H', 10)

def printInOrder(root, height, preStr, length):
    if not root:
        return
    printInOrder(root.right, height+1, 'v', length)
    string = preStr + root.val + preStr
    leftLen = (length - len(string)) // 2
    rightLen = length - len(string)- leftLen
    res = " "*leftLen + string + " "*rightLen
    print(" "*height*length + res)
    printInOrder(root.left, height+1, '^', length)

#先序遍历（递归）
def pre_order(root):
    if root:
        print(root.val, end=' ')
        pre_order(root.left)
        pre_order(root.right)

#中序遍历（递归）
def mid_order(root):
    if root:
        mid_order(root.left)
        print(root.val, end=' ')
        mid_order(root.right)

#后序遍历（递归）
def post_order(root):
    if root:
        post_order(root.left)
        post_order(root.right)
        print(root.val, end=' ')

#先序遍历（非递归）
def preorder(root):
    if not root:
        return
    stack = []
    while root or len(stack):
        if root:
            stack.append(root)
            print(root.val, end=' ')
            root = root.left
        else:
            root = stack.pop()
            root = root.right

#中序遍历（非递归）
def midorder(root):
    if not root:
        return
    stack = []
    while root or stack:
        if root:
            stack.append(root)
            root = root.left
        else:
            root = stack.pop()
            print(root.val, end=' ')
            root = root.right

#后序遍历（非递归）
def postorder(root):
    if not root:
        return
    stack1 = []
    stack2 = []
    while root or stack1:
        if root:
            stack1.append(root)
            stack2.append(root.val)
            root = root.right
        else:
            root = stack1.pop()
            root = root.left
    while stack2:
        print(stack2.pop(), end=' ')

#后序遍历（非递归方法2）
def postorder2(root):
    if not root:
        return
    stack = []
    stack.append(root)
    c = None
    while stack:
        c = stack[-1]
        if c.left and c.left != root and c.right != root:
            stack.append(c.left)
        elif c.right and c.right != root:
            stack.append(c.right)
        else:
            print(stack.pop().val, end=' ')
            root = c

#Morris前序遍历
def morrisPre(root):
    if not root:
        return
    while root:
        cur2 = root.left
        if cur2:
            while cur2.right and cur2.right != root:
                cur2 = cur2.right
            if not cur2.right:
                cur2.right = root
                print(root.val, end=' ')
                root = root.left
                continue
            else:
                cur2.right = None
        else:
            print(root.val, end=' ')
        root = root.right

#Morris中序遍历
def morrisIn(root):
    if not root:
        return
    while root:
        cur2 = root.left
        if cur2:
            while cur2.right and cur2.right != root:
                cur2 = cur2.right
            if not cur2.right:
                cur2.right = root
                root = root.left
                continue
            else:
                cur2.right = None
        print(root.val, end=' ')
        root = root.right

#Morris后序遍历
def morrisPost(root):
    def printRightEdge(root):
        tail = reverseEdge(root)
        cur = tail
        while cur:
            print(cur.val, end=' ')
            cur = cur.right
        reverseEdge(tail)

    def reverseEdge(root):
        pre = None
        while root:
            next = root.right
            root.right = pre
            pre = root
            root = next
        return pre

    if not root:
        return
    head = root
    while root:
        cur = root.left
        if cur:
            while cur.right and cur.right != root:
                cur = cur.right
            if not cur.right:
                cur.right = root
                root = root.left
                continue
            else:
                cur.right = None
                printRightEdge(root.left)
        root = root.right
    printRightEdge(head)

#层次遍历
def levelorder(root):
    if not root:
        return
    queue = []
    queue.append(root)
    while queue:
        root = queue.pop(0)
        print(root.val, end=' ')
        if root.left:
            queue.append(root.left)
        if root.right:
            queue.append(root.right)

#二叉树的序列化和反序列化（先序）
def serialByPre(root):
    if not root:
        return '#!'
    res = root.val + '!'
    res += serialByPre(root.left)
    res += serialByPre(root.right)
    return res

def reconByPreString(preStr):
    def reconPreOrder(values):
        key = values.pop(0)
        if key == '#':
            return None
        root = TreeNode(key)
        root.left = reconPreOrder(values)
        root.right = reconPreOrder(values)
        return root

    values = preStr.split('!')
    return reconPreOrder(values)

#二叉树的序列化和反序列化（层次）
def serialByLevel(root):
    if root == '#':
        return '#!'
    stack = []
    stack.append(root)
    res = root.val + '!'
    while stack:
        root = stack.pop()
        if root.left:
            res += root.left.val + '!'
            stack.append(root.left)
        else:
            res += '#!'
        if root.right:
            res += root.right.val + '!'
            stack.append(root.right)
        else:
            res += '#!'
    return res

def reconByLevelString(levStr):
    def generateNode(key):
        if key == '#':
            return None
        return TreeNode(key)

    values = levStr.split('!')
    head = generateNode(values.pop(0))
    queue = []
    if head:
        queue.append(head)
    while queue:
        root = queue.pop(0)
        root.left = generateNode(values.pop(0))
        root.right = generateNode(values.pop(0))
        if root.left:
            queue.append(root.left)
        if root.right:
            queue.append(root.right)
    return head

#树的深度
def getDepth(root):
    if not root:
        return 0
    return 1 + max(getDepth(root.left), getDepth(root.right))

#树的带权路径长度
def getWPL(root, depth=0):
    if not root:
        return 0
    if not root.left and not root.right:
        return depth * int(root.val)
    return getWPL(root.left, depth + 1) + getWPL(root.right, depth + 1)

#在二叉树中找到累加和为指定值的最长路径长度
def getMaxLength(root, K):
    def getLengthByPreOrder(root, K, preSum, level, length, map):
        if not root:
            return length
        curSum = preSum + int(root.val)
        if curSum not in map:
            map[curSum] = level
        if curSum-K in map:
            length = max(level - map.get(curSum-K), length)
        length = getLengthByPreOrder(root.left, K, curSum, level+1, length, map)
        length = getLengthByPreOrder(root.right, K, curSum, level+1, length, map)
        if level == map.get(curSum):
            map.pop(curSum)
        return length

    if not root:
        return
    map = {}
    map[0] = 0
    return getLengthByPreOrder(root, K, 0, 1, 0, map)

#找到二叉树中的最大搜索二叉树
def biggestSubBST(root):
    def findBiggestSubBST(root, record):
        if not root:
            record[0] = 0
            record[1] = sys.maxsize
            record[2] = -sys.maxsize
            return None
        leftBST = findBiggestSubBST(root.left, record)
        leftSize = record[0]
        leftMin = record[1]
        leftMax = record[2]
        rightBST = findBiggestSubBST(root.right, record)
        rightSize = record[0]
        rightMin = record[1]
        rightMax = record[2]
        record[1] = min(leftMin, int(root.val))
        record[2] = max(rightMax, int(root.val))
        if leftBST == root.left and rightBST == root.right and \
                rightMin > int(root.val) and leftMax < int(root.val):
            record[0] = leftSize + rightSize + 1
            return root
        record[0] = max(leftSize, rightSize)
        if leftSize > rightSize:
            return leftBST
        else:
            return rightBST

    if not root:
        return
    record = [0 for i in range(3)]
    return findBiggestSubBST(root, record)

#二叉树中符合搜索二叉树条件的最大拓扑结构的长度
def bstTopoSize(root):
    def maxTopo(root, head):
        if root and head and isSubNode(root,head):
            return maxTopo(root, head.left) + maxTopo(root, head.right) + 1
        return 0

    def isSubNode(root, head):
        if not root:
            return False
        if root == head:
            return True
        if root.val > head.val:
            return isSubNode(root.left, head)
        if root.val < head.val:
            return isSubNode(root.right, head)

    if not root:
        return 0
    maxSize = maxTopo(root,root)
    maxSize = max(bstTopoSize(root.left), maxSize)
    maxSize = max(bstTopoSize(root.right), maxSize)
    return maxSize

#二叉树的按层打印与ZigZag打印
def printByLevel(root):
    if not root:
        return
    print("Print binary tree by level")
    queue = []
    queue.append(root)
    last = root
    level = 1
    print("Level " + str(level) + ':', end=' ')
    while queue:
        root = queue.pop(0)
        print(root.val, end=' ')
        if root.left:
            nlast = root.left
            queue.append(root.left)
        if root.right:
            nlast = root.right
            queue.append(root.right)
        if root == last and queue:
            last = nlast
            print()
            level += 1
            print("Level " + str(level) + ":", end=' ')

def printByZigZag(root):
    if not root:
        return
    print("Print binary tree by ZigZag")
    deque = []
    deque.append(root)
    islr = True
    last = root
    nlast = None
    level = 1
    print("Level " + str(level) + ":", end=' ')
    while deque:
        if islr:
            root = deque.pop(0)
            print(root.val, end=' ')
            if root.left:
                if nlast == None:
                    nlast = root.left
                deque.append(root.left)
            if root.right:
                if nlast == None:
                    nlast = root.right
                deque.append(root.right)
        else:
            root = deque.pop()
            print(root.val, end=' ')
            if root.right:
                if nlast == None:
                    nlast = root.right
                deque.insert(0, root.right)
            if root.left:
                if nlast == None:
                    nlast = root.left
                deque.insert(0, root.left)
        if root == last and deque:
            islr = not islr
            last = nlast
            nlast = None
            print()
            level += 1
            print("Level " + str(level) + ":", end=' ')

#找到搜索二叉树中两个错误的节点
def getTwoErrorNode(root):
    errs = [None for i in range(2)]
    if not root:
        return errs
    stack = []
    pre = None
    while root or stack:
        if root:
            stack.append(root)
            root = root.left
        else:
            root = stack.pop()
            if pre and pre.val > root.val:
                if errs[0] == None:
                    errs[0] = pre
                errs[1] = root
            pre = root
            root = root.right
    return errs

#判断t1树是否包含t2树全部的拓扑结构
def isContain(t1, t2):
    def check(t1, t2):
        if not t2:
            return True
        if not t1 or t1.val != t2.val:
            return False
        return check(t1.left, t2.left) and check(t1.right, t2.right)

    return check(t1, t2) or isContain(t1.left, t2) or isContain(t1.right, t2)

#判断t1树中是否有与t2树拓扑结构完全相同的子树
def isSubTopoBST(t1, t2):
    def serialTreeByPre(root):
        if not root:
            return '#!'
        res = root.val + '!'
        res += serialTreeByPre(root.left)
        res += serialTreeByPre(root.right)
        return res

    def isTopoBST(s, m):
        if not s or not m or len(s) < len(m) or len(m) < 1:
            return False
        mi = 0
        si = 0
        nextarr = getNextArray(m)
        while mi < len(m) and si < len(s):
            if s[si] == m[mi]:
                mi += 1
                si += 1
            elif nextarr[mi] == -1:
                si += 1
            else:
                mi = nextarr[mi]
        if mi == len(m):
            return True
        else:
            return False

    def getNextArray(m):
        if len(m) == 1:
            return -1
        nextarr = [None for i in range(len(m))]
        nextarr[0] = -1
        nextarr[1] = 0
        pos = 2
        cn = 0
        while pos < len(m):
            if m[pos-1] == m[cn]:
                nextarr[pos] = cn + 1
                pos += 1
                cn += 1
            elif cn > 0:
                cn = nextarr[cn]
            else:
                nextarr[pos] = 0
                pos += 1
        return nextarr

    if not t1 or not t2:
        return False
    str1 = serialTreeByPre(t1)
    str2 = serialTreeByPre(t2)
    arr1 = str1.split('!')
    arr1.pop()
    arr2 = str2.split('!')
    arr2.pop()
    return isTopoBST(arr1, arr2)

def isSubTopoBST2(t1, t2):
    def isTopoBST2(t1, t2):
        if not t1 and not t2:
            return True
        if not t1 or not t2 or t1.val != t2.val:
            return False
        return isTopoBST2(t1.left, t2.left) and isTopoBST2(t1.right, t2.right)

    if not t1 or not t2:
        return False
    return isTopoBST2(t1, t2) or isSubTopoBST2(t1.left, t2) or isSubTopoBST2(t1.right, t2)

#判断二叉树是否为平衡二叉树
def isBalance(root):
    def judgeIsBalance(root, level, res):
        if not root:
            return level
        lH = judgeIsBalance(root.left , level+1, res)
        if res[0] == False:
            return level
        rH = judgeIsBalance(root.right , level+1, res)
        if res[0] == False:
            return level
        if abs(lH - rH) > 1:
            res[0] = False
        return max(lH, rH)

    print("该二叉树是否为平衡二叉树？")
    if not root:
        return True
    res = [True]
    judgeIsBalance(root, 1, res)
    return res[0]

#判断数组是否为某搜索二叉树的后序遍历的结果
def isPostArray(arr):
    def isPost(arr, start, end):
        if start == end:
            return True
        leftEnd = None
        rightStart = None
        for i in range(end):
            if arr[i] < arr[end]:
                leftEnd = i
        for i in range(end-1, -1, -1):
            if arr[i] > arr[end]:
                rightStart = i
        if not leftEnd or not rightStart:
            return isPost(arr, start, end-1)
        if leftEnd != rightStart-1:
            return False
        return isPost(arr, start, leftEnd) and isPost(arr, rightStart, end-1)

    print("该数组是否为某搜索二叉树后序遍历的结果？")
    if arr == None or len(arr) == 0:
        return False
    return isPost(arr, 0, len(arr)-1)

#根据后序数组重建搜索二叉树
def postArrayToBST(arr):
    def arrayToBST(arr, start, end):
        if start > end:
            return None
        root = TreeNode(arr[end])
        leftEnd = -1
        rightStart = end
        for i in range(end):
            if arr[i] < arr[end]:
                leftEnd = i
        for i in range(end-1, -1, -1):
            if arr[i] > arr[end]:
                rightStart = i
        root.left = arrayToBST(arr, start, leftEnd)
        root.right = arrayToBST(arr, rightStart, end-1)
        return root

    if arr == None:
        return None
    return arrayToBST(arr, 0, len(arr)-1)

#判断一棵树是否为搜索二叉树
def isBST(root):
    print("判断一棵树是否为搜索二叉树：")
    if not root:
        return True
    res = True
    pre = None
    cur1 = root
    cur2 = None
    while cur1:
        cur2 = cur1.left
        if cur2:
            while cur2.right and cur2.right != cur1:
                cur2 = cur2.right
            if cur2.right == None:
                cur2.right = cur1
                cur1 = cur1.left
                continue
            else:
                cur2.right = None
        if pre and int(pre.val) > int(cur1.val):
            res = False
        pre = cur1
        cur1 = cur1.right
    return res

#判断一棵树是否为完全二叉树
def isCBT(root):
    print("该树是否为完全二叉树：")
    if not root:
        return True
    isLeaf = False
    queue = []
    queue.append(root)
    while queue:
        root = queue.pop(0)
        left = root.left
        right = root.right
        if (not left and right) or (isLeaf and (left  or right)):
            return False
        if left:
            queue.append(left)
        if right:
            queue.append(right)
        else:
            isLeaf = True
    return True

#通过有序数组生成平衡搜索二叉树
def generateTree(arr):
    def generate(arr, start, end):
        if start > end:
            return None
        center = (start + end) // 2
        head = TreeNode(arr[center])
        head.left = generate(arr, start, center-1)
        head.right = generate(arr, center+1, end)
        return head

    print("通过有序数组生成平衡二叉搜索树")
    if not arr or len(arr) == 0:
        return None
    return generate(arr, 0, len(arr)-1)

#在二叉树中找到一个节点的后继节点
def getNextNode(node):
    if not Node:
        return
    if node.right:
        node = node.right
        if node.left:
            while node.left:
                node = node.left
        return node
    parent = node.parent
    while parent and parent.left != node:
        node = parent
        parent = node.parent
    return parent

#在二叉树中找到两个节点的最近公共节点
def lowestAncestor(head, o1, o2):
    if head == None or head == o1 or head == o2:
        return head
    left = lowestAncestor(head.left, o1, o2)
    right = lowsetAncestor(head.right, o1, o2)
    if left and right:
        return head
    return left if left else right

#二叉树节点间的最大距离
def maxDistance(head):
    def posOrder(head, record):
        if head == None:
            record[0] = 0
            return 0
        leftMax = posOrder(head.left, record)
        maxfromLeft = record[0]
        rightMax = posOrder(head.right, record)
        maxfromRight = record[0]
        record[0] = max(maxfromLeft, maxfromRight) + 1
        curMax = maxfromLeft + maxfromRight + 1
        return max(max(maxfromLeft, maxfromRight),curMax)

    print("二叉树节点间的最大距离")
    if head == None:
        return 0
    record = [None]
    return posOrder(head, record)

#先序，中序，和后序数组两两结合重构二叉树

#先序，中序数组重构二叉树
def preInToTree(pre, mid):
    def preIn(pre, pi, pj, mid, mi, mj, map):
        if pi > pj:
            return None
        head = TreeNode(pre[pi])
        index = map.get(pre[pi])
        head.left = preIn(pre, pi+1, pi+index-mi, mid, mi, index-1, map)
        head.right = preIn(pre, pi+index-mi+1, pj, mid, index+1, mj, map)
        return head

    print("先序，中序数组重构二叉树")
    if pre == None or mid == None or len(pre) != len(mid):
        return None
    map = {}
    for i in range(len(mid)):
        map[mid[i]] = i
    return preIn(pre, 0, len(pre)-1, mid, 0, len(mid)-1, map)

#中序，后序数组重构二叉树
def inPosToTree(mid, pos):
    def inPos(mid, mi, mj, pos, si, sj, map):
        if si > sj:
            return None
        head = TreeNode(pos[sj])
        index = map.get(pos[sj])
        head.left = inPos(mid, mi, index-1, pos, si, si+index-mi-1, map)
        head.right = inPos(mid, index+1, mj, pos, si+index-mi, sj-1, map)
        return head
    
    print("中序，后序数组重构二叉树")
    if mid == None or pos == None or len(mid) != len(pos):
        return None
    map = {}
    for i in range(len(mid)):
        map[mid[i]] = i
    return inPos(mid, 0, len(mid)-1, pos, 0, len(pos)-1, map)

#先序，后序数组重构二叉树
#一棵二叉树除叶子节点外，其他所有的节点都有左孩子和右孩子，才能被先序和后序数组重构出来
def prePosToTree(pre, pos):
    def prePos(pre, pi, pj, pos, si, sj, map):
        head = TreeNode(pre[pi])
        if pi == pj:
            return head
        pi += 1
        index = map.get(pre[pi])
        head.left = prePos(pre, pi, pi+index-si, pos, si, index, map)
        head.right = prePos(pre, pi+index-si+1, pj, pos, index+1, sj-1, map)
        return head

    print("先序，后序数组重构二叉树")
    if pre == None or pos == None or len(pre) != len(pos):
        return None
    map = {}
    for i in range(len(pos)):
        map[pos[i]] = i
    return prePos(pre, 0, len(pre)-1, pos, 0, len(pos)-1, map)

#通过先序和中序数组生成后序数组
def preInToPos(pre, mid):
    def preInPos(pre, pi, pj, mid, mi, mj, pos, sj, map):
        if pi > pj:
            return sj
        pos[sj] = pre[pi]
        sj -= 1
        index = map.get(pre[pi])
        sj = preInPos(pre, pi+index-mi+1, pj, mid, index+1, mj, pos, sj, map)
        return preInPos(pre, pi+1, pi+index-mi, mid, mi, index-1, pos, sj, map)

    print("通过先序和中序数组生成后序数组")
    if pre == None or mid == None:
        return []
    pos = [None for i in range(len(mid))]
    map = {}
    for i in range(len(mid)):
        map[mid[i]] = i
    preInPos(pre, 0, len(pre)-1, mid, 0, len(mid)-1, pos, len(mid)-1, map)
    return pos

#统计所有可能的二叉树结构的种数
def allNum(n):
    if n < 2:
        return 1
    num = [0 for i in range(n+1)]
    num[0] = 1
    for i in range(1, n+1):
        for j in range(1, i+1):
            num[i] += num[j-1] * num[i-j]
    return num[n]

#统计完全二叉树的节点个数
def nodeNum(head):
    def getHeight(head, level):
        while head:
            level += 1
            head = head.left
        return level-1

    def bs(head, level, height):
        if level == height:
            return 1
        h = getHeight(head.right, level+1)
        if h == height:
            return (1 << (height-level)) + bs(head.right, level+1, height)
        else:
            return (1 << (height-level-1)) + bs(head.left, level+1, height)


    print("统计完全二叉树的节点个数")
    if head == None:
        return 0
    height = getHeight(head, 1)
    return bs(head, 1, height)


t = None
arr = ['6','1','0','#','#','3','#','#','12','10','4','2','#','#','5','#','#','14','11','#','#','15','#','#','13','20','#','#','16','#','#']
test = ['4','5','1','#','#','3','#','#','2','#','#']
test1 = ['1','2','4','8','#','#','9','#','#','5','10','#','#','#','3','6','#','#','7','#','#']
test2 = ['2','4','8','#','#','#','5','#','#']
list1 = ['1','2','4','#','8','#','#','5','9','#','#','#','3','6','#','#','7','#','#']
list2 = ['9','4','5','#','#','8','#','#','10','#','#']
arr1 = ['1','2','3','4','5','6','7','8']
prearr = ['1','2','4','5','8','9','3','6','7']
midarr = ['4','2','8','5','9','1','6','3','7']
posarr = ['4','8','9','5','2','6','7','3','1']
test3 = ['1','2','4','#','#','5','#','#','3','6','#','#','7','#','#']
print(allNum(7))
print(preInToPos(prearr, midarr))
preIn = preInToTree(prearr, midarr)
printTree(preIn)
inPos = inPosToTree(midarr, posarr)
printTree(inPos)
prePos = prePosToTree(prearr, posarr)
printTree(prePos)
t3 = postArrayToBST(arr1)
t4 = generateTree(arr1)
printTree(t3)
printTree(t4)
print(isPostArray(arr1))
t1 = createTreeByPreOrder(test3)
t2 = createTreeByPreOrder(list2)
print(nodeNum(t1))
print(maxDistance(t1))
print(isCBT(t2))
print(isBST(t2))
print(isBalance(list2))
print(isSubTopoBST2(list1, list2))
t = createTreeByPreOrder(test1)
test1 = createTreeByPreOrder(test2)
print(isContain(t, test1))
printTree(t)
print(getMaxLength(t, -9))
biggestSubBst = biggestSubBST(t)
print(biggestSubBst.val)
print(bstTopoSize(t))
printByLevel(t)
print()
printByZigZag(t)
print()
print([element.val for element in getTwoErrorNode(t)])
print("先序遍历（递归）: ")
pre_order(t)
print()
print("中序遍历（递归）：")
mid_order(t2)
print()
print("后序遍历（递归）：")
post_order(t)
print()
print("先序遍历（非递归）: ")
preorder(t)
print()
print("中序遍历（非递归）：")
midorder(t)
print()
print("后序遍历（非递归）：")
postorder2(t)
print()
print("先序遍历（Morris）: ")
morrisPre(t)
print()
print("中序遍历（Morris）：")
morrisIn(t)
print()
print("后序遍历（Morris）：")
morrisPost(t)
print()
print("层次遍历：")
levelorder(t)
print()
strPre = serialByPre(t)
print("序列化二叉树(先序)：" + strPre)
root = reconByPreString(strPre)
print("反序列化二叉树(先序)：")
morrisPre(root)
print()
res = serialByLevel(t)
print("序列化二叉树(层次)：" + res)
head = reconByLevelString(res)
print("反序列化二叉树(层次)：")
levelorder(head)
print()
print("树的深度：")
print(getDepth(t))
print("树的带权路径长度：")
print(getWPL(t))
