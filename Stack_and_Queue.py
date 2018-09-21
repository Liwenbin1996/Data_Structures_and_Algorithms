#设计一个有getMin功能的栈
class NewStack1:
    def __init__(self):
        self.stackData = []
        self.stackMin = []

    def push(self, newNum):
        self.stackData.append(newNum)
        if len(self.stackMin) == 0 or newNum <= self.getMin():
            self.stackMin.append(newNum)
    
    def pop(self):
        if len(self.stackData) == 0:
            raise Exception("stack is empty!")
        value = self.stackData.pop()
        if self.getMin() == value:
            self.stackMin.pop()
        return value

    def getMin(self):
        if len(self.stackMin) == 0:
            raise Exception("stack is empty!")
        return self.stackMin[-1]


class NewStack2:
    def __init__(self):
        self.stackData = []
        self.stackMin = []

    def push(self, newNum):
        self.stackData.append(newNum)
        if len(self.stackMin) == 0 or newNum < self.getMin():
            self.stackMin.append(newNum)
        else:
            self.stackMin.append(self.getMin())

    def pop(self):
        if len(self.stackData) == 0:
            raise Exception("Stack is empty!")
        self.stackMin.pop()
        return self.stackData.pop()

    def getMin(self):
        if len(self.stackMin) == 0:
            raise Exception("Stack is empty!")
        return self.stackMin[-1]


#由两个栈组成队列
class TwoStackQueue:
    stackPush = []
    stackPop = []

    def add(self, newNum):
        self.stackPush.append(newNum)

    def poll(self):
        if not self.stackPush and not self.stackPop:
            raise Exception("Queue is empty!")
        elif not self.stackPop:
            while self.stackPush:
                self.stackPop.append(self.stackPush.pop())
        return self.stackPop.pop()

    def peek(self):
        if not self.stackPush and not self.stackPop:
            raise Exception("Queue is empty!")
        elif not self.stackPop:
            while self.stackPush:
                self.stackPop.append(self.stackPush.pop())
        return self.stackPop[-1]


#如何仅用递归函数和栈操作逆序一个栈
def reverse(stack):
    def getAndRemoveLast(stack):
        result = stack.pop()
        if len(stack) == 0:
            return result
        else:
            i = getAndRemoveLast(stack)
            stack.append(result)
            return i

    if len(stack) == 0:
        return
    i = getAndRemoveLast(stack)
    reverse(stack)
    stack.append(i)
    return stack


#用一个栈实现另一个栈的排序
def sortByStack(stack):
    if len(stack) < 2:
        return stack
    stack1 = []
    while stack:
        cur = stack.pop()
        if len(stack1) == 0 or stack1[-1] >= cur:
            stack1.append(cur)
        else:
            while stack1:
                stack.append(stack1.pop())
            stack1.append(cur)
    while stack1:
        stack.append(stack1.pop())
    return stack


#生成窗口最大值数组
def getMaxWindow(arr, w):
    if arr == None or w < 1 or len(arr) < w:
        return
    deque = []
    res = []
    for i in range(len(arr)):
        while deque and arr[deque[-1]] <= arr[i]:
            deque.pop()
        deque.append(i)
        if deque[0] == i - w:
            deque.pop(0)
        if i-w+1 >= 0:
            res.append(arr[deque[0]])
    return res


#构造数组的MaxTree
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def getMaxTree(arr):
    nArr = [Node(arr[i]) for i in range(len(arr))]
    lBigMap = {}
    rBigMap = {}
    stack = []
    for i in range(len(nArr)):
        curNode = nArr[i]
        while stack and stack[-1].value < curNode.value:
            cur = stack.pop()
            lBigMap[cur] = stack[-1] if stack else None
            rBigMap[cur] = curNode
        stack.append(curNode)
    while stack:
        cur = stack.pop()
        lBigMap[cur] = stack[-1] if stack else None
        rBigMap[cur] = None
#    for i in range(len(nArr)-1, -1, -1):
#        curNode = nArr[i]
#        while stack and stack[-1].value < curNode.value:
#            cur = stack.pop()
#            rBigMap[cur] = stack[-1] if stack else None
#        stack.append(curNode)
#    while stack:
#        cur = stack.pop()
#        rBigMap[cur] = stack[-1] if stack else None
    head = None
    for i in range(len(nArr)):
        curNode = nArr[i]
        left = lBigMap[curNode]
        right = rBigMap[curNode]
        if left == None and right == None:
            head = curNode
        elif left == None:
            if right.left == None:
                right.left = curNode
            else:
                right.right = curNode
        elif right == None:
            if left.left == None:
                left.left = curNode
            else:
                left.right = curNode
        else:
            parent = left if left.value < right.value else right
            if parent.left == None:
                parent.left = curNode
            else:
                parent.right = curNode
    return head
        

def preOrderMaxTree(head):
    if head == None:
        return
    print(head.value, end=' ')
    preOrderMaxTree(head.left)
    preOrderMaxTree(head.right)


#求最大子矩阵的大小
def maxRecSize(map):
    def maxRecFromBottom(height):
        if height == None or len(height) == 0:
            return 0
        stack = []
        maxArea = 0
        for i in range(len(height)):
            while stack and height[stack[-1]] >= height[i]:
                j = stack.pop()
                k = stack[-1] if stack else -1
                maxArea = max(maxArea, (i-k-1) * height[j])
            stack.append(i)
        while stack:
            j = stack.pop()
            k = stack[-1] if stack else -1
            maxArea = max(maxArea, (len(height)-k-1) * height[j])
        return maxArea

    if map == None or len(map) == 0 or len(map[0]) == 0:
        return 0
    height = [0 for i in range(len(map[0]))]
    maxArea = 0
    for i in range(len(map)):
        for j in range(len(map[0])):
            height[j] = 0 if map[i][j] == 0 else height[j] + 1
        maxArea = max(maxArea, maxRecFromBottom(height))
    return maxArea


#最大值减去最小值小于或等于num的子数组数量
def getNum(arr, num):
    if arr == None or len(arr) == 0:
        return 0
    qmin = []
    qmax = []
    i = 0
    j = 0
    res = 0
    while i < len(arr):
        while j < len(arr):
            while qmin and arr[qmin[-1]] >= arr[j]:
                qmin.pop()
            qmin.append(j)
            while qmax and arr[qmax[-1]] <= arr[j]:
                qmax.pop()
            qmax.append(j)
            if arr[qmax[0]] - arr[qmin[0]] > num:
                break
            j += 1
        if qmin[0] == i:
            qmin.pop(0)
        if qmax[0] == i:
            qmax.pop(0)
        res += j - i
        i += 1
    return res








#stack = NewStack2()
#stack.push(1)
#stack.push(2)
#print(stack.getMin())
#print(stack.pop())
#print(stack.pop())
#queue = TwoStackQueue()
#queue.add(3)
#queue.add(4)
#print(queue.peek())
#print(queue.poll())
#print(queue.peek())
#print(queue.poll())
#print(reverse([1,2,3,4,5]))
#print(sortByStack([1,3,4,4,5,2]))
#print(getMaxWindow([4,3,5,4,3,3,6,7], 3))
#preOrderMaxTree(getMaxTree([3,4,5,1,2]))
#print()
#print(maxRecSize([[1,0,1,1], [1,1,1,1], [1,1,1,0]]))
print(getNum([0,3,6,9], 2))
