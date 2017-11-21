import math
import random

class Node:
    def __init__(self, val=None):
        self.val = val
        self.next = None

class DoubleNode:
    def __init__(self, val = None):
        self.val = val
        self.pre = None
        self.next = None


#打印两个有序链表的公共部分
def printCommonPart(head1, head2):
    if head1 == None or head2 == None:
        return
    print("Common Part:", end=' ')
    while head1 != None and head2 != None:
        if head1.val > head2.val:
            head2 = head2.next
        elif head1.val < head2.val:
            head1 = head1.next
        else:
            print(head1.val, end=' ')
            head1 = head1.next
            head2 = head2.next
    print()


#在单链表和双链表中删除倒数第K个节点
def removeLastKthNode1(head, k):
    if head == None or k < 1:
        return head
    cur = head
    while cur != None:
        k -= 1
        cur = cur.next
    if k == 0:
        return head.next
    elif k < 0:
        cur = head
        while k+1 != 0:
            cur = cur.next
            k += 1
        cur.next = cur.next.next
    return head


def removeLastKthNode2(head, k):
    if head == None or k < 1:
        return head
    fast = slow = head
    while k > 0:
        k -= 1
        if fast == None:
            return head
        else:
            fast = fast.next
    if fast == None:
        return head.next
    while fast.next != None:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return head


def removeLastKthDoubleNode1(head, k):
    if head == None or k < 1:
        return head
    cur = head
    while cur != None:
        k -= 1
        cur = cur.next
    if k == 0:
        head = head.next
        head.pre = None
    elif k < 0:
        cur = head
        while k != -1:
            k += 1
            cur = cur.next
        cur.next = cur.next.next
        if cur.next != None:
            cur.next.pre = cur
    return head

def removeLastKthDoubleNode2(head, k):
    if head == None or k < 1:
        return head
    fast = slow = head
    while k > 0:
        k -= 1
        if fast != None:
            fast = fast.next
        else:
            return head
    if fast == None:
        head = head.next
        head.pre = None
    else:
        while fast.next != None:
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next
        if slow.next != None:
            slow.next.pre = slow
    return head


#删除链表的中间节点和a/b处的节点
def removeMidNode(head):
    if head == None or head.next == None:
        return head
    if head.next.next == None:
        return head.next
    pre = head
    cur = head.next.next
    while cur.next != None and cur.next.next != None:
        pre = pre.next
        cur = cur.next.next
    pre.next = pre.next.next
    return head

def removeByRatio(head, a, b):
    if head == None or a < 1 or a > b:
        return head
    n = 0
    cur = head
    while cur != None:
        cur = cur.next
        n += 1
    n = math.ceil(a / b * n)
    if n == 1:
        return head.next
    cur = head
    while n-1 != 1:
        cur = cur.next
        n -= 1
    cur.next = cur.next.next
    return head


#反转单向和双向链表
def reverseList(head):
    if head == None:
        return
    pre = None
    while head != None:
        next = head.next
        head.next = pre
        pre = head
        head = next
    return pre

def reverseDoubleList(head):
    if head == None:
        return
    pre = None
    while head != None:
        next = head.next
        head.next = pre
        head.pre = next
        pre = head
        head = next
    return pre


#反转部分单向链表
def reversePart(head, start, end):
    if head == None or head.next == None:
        return head
    length = 0
    pre = None
    pos = None
    node1 = head
    while node1 != None:
        length += 1
        pre = node1 if length == start-1 else pre
        pos = node1 if length == end+1 else pos
        node1 = node1.next
    if start >= end or start < 1 or end > length:
        return head
    node1 = pre.next if pre != None else head
    node2 = node1.next
    node1.next = pos
    while node2 != pos:
        next = node2.next
        node2.next = node1
        node1 = node2
        node2 = next
    if pre != None:
        pre.next = node1
        return head
    return node1


#环形单链表的约瑟夫问题
def josephusKill1(head, m):
    if head == None or head.next == None or m < 1:
        return head
    pre = head
    while pre.next != head:
        pre = pre.next
    count = 1
    while head != pre:
        if count != m:
            head = head.next
            pre = pre.next
            count += 1
        else:
            pre.next = head.next
            head = pre.next
            count = 1
    return head


def josephusKill2(head, m):
    def getLive(n, m):
        if n == 1:
            return 1
        return (getLive(n-1, m) + m - 1) % n + 1

    if head == None or head.next == None or m < 1:
        return head
    n = 1
    cur = head
    while cur.next != head:
        n += 1
        cur = cur.next
    n = getLive(n, m)
    while n-1 != 0:
        n -= 1
        head = head.next
    head.next = head
    return head


#判断一个链表是否为回文结构
def isPalindrome1(head):
    if head == None or head.next == None:
        return True
    stack = []
    cur = head
    while cur != None:
        stack.append(cur)
        cur = cur.next
    while stack:
        if stack.pop().val != head.val:
            return False
        head = head.next
    return True


def isPalindrome2(head):
    if head == None or head.next == None:
        return True
    stack = []
    pre = head
    cur = head
    while cur.next != None and cur.next.next != None:
        pre = pre.next
        cur = cur.next.next
    while pre != None:
        stack.append(pre)
        pre = pre.next
    while stack:
        if stack.pop().val != head.val:
            return False
        head = head.next
    return True


def isPalindrome3(head):
    if head == None or head.next == None:
        return True
    pre = head
    cur = head
    while cur.next != None and cur.next.next != None:
        pre = pre.next 
        cur = cur.next.next
    node = pre.next
    pre.next = None 
    while node != None:
        next = node.next
        node.next = pre
        pre = node
        node = next
    node = pre
    res = True
    while pre != None and head != None:
        if pre.val != head.val:
            res = False
            break
        pre = pre.next
        head = head.next
    pre = node.next
    node.next = None
    while pre != None:
        next = pre.next
        pre.next = node
        node = pre
        pre = next
    return res


#将单向链表按某值划分成左边小，中间相等，右边大的形式
def listPartition(head, pivot):
    def partition(nodeArr, pivot):
        left = -1
        right = len(nodeArr)
        index = 0
        while index < right:
            if nodeArr[index].val < pivot:
                left += 1
                nodeArr[left], nodeArr[index] = nodeArr[index], nodeArr[left]
                index += 1
            elif nodeArr[index].val == pivot:
                index += 1
            else:
                right -= 1
                nodeArr[index], nodeArr[right] = nodeArr[right], nodeArr[index]
            


    if head == None or head.next == None:
        return head
    cur = head
    n = 0
    while cur != None:
        n += 1
        cur = cur.next
    nodeArr = []
    cur = head
    while cur != None:
        nodeArr.append(cur)
        cur = cur.next
    partition(nodeArr, pivot)
    for i in range(n-1):
        nodeArr[i].next = nodeArr[i+1]
    nodeArr[-1].next = None
    return nodeArr[0]


def listPartition2(head, pivot):
    if head == None or head.next == None:
        return head
    sH = None
    sT = None
    eH = None
    eT = None 
    bH = None 
    bT = None
    while head != None:
        next = head.next
        head.next = None
        if head.val < pivot:
            if sH == None:
                sH = head
                sT = head
            else:
                sT.next = head
                sT = head
        elif head.val == pivot:
            if eH == None:
                eH = head
                eT = head
            else:
                eT.next = head
                eT = head
        else:
            if bH == None:
                bH = head
                bT = head
            else:
                bT.next = head
                bT = head
        head = next
    head = None
    if sT != None:
        head = sH
        if eH != None:
            sT.next = eH
        elif bH != None:
            sT.next = bH
    if eT != None:
        head = head if head != None else eH
        if bH != None:
            eT.next = bH
    return head


#复制含有随机指针节点的链表
class RandNode:
    def __init__(self, data):
        self.val = data
        self.next = None
        self.rand = None

def copyListWithRand1(head):
    if head == None:
        return None
    map = {}
    cur = head
    while cur != None:
        map[cur] = RandNode(cur.val)
        cur = cur.next
    cur = head
    while cur != None:
        map[cur].next = None if cur.next == None else map[cur.next]
        map[cur].rand = None if cur.rand == None else map[cur.rand]
        cur = cur.next
    return map[head] 
    
    
def copyListWithRand2(head):
    if head == None:
        return None
    cur = head
    while cur != None:
        next = cur.next
        cur.next = RandNode(cur.val)
        cur.next.next = next
        cur = next
    cur = head
    while cur != None:
        cur.next.rand = None if cur.rand == None else cur.rand.next
        cur = cur.next.next
    copyHead = head.next
    cur = head
    while cur != None:
        next = cur.next
        cur.next = next.next
        next.next = None if next.next == None else next.next.next
        cur = cur.next
    return copyHead
    

#两个单链表生成相加链表
def addList1(head1, head2):
    if head1 == None or head2 == None:
        raise Exception("Input Error!")
    s1 = []
    s2 = []
    while head1 != None:
        s1.append(head1.val)
        head1 = head1.next
    while head2 != None:
        s2.append(head2.val)
        head2 = head2.next
    print("post")
    carry = 0
    pre = None
    while s1 or s2:
        num1 = 0 if not s1 else s1.pop()
        num2 = 0 if not s2 else s2.pop()
        sum = num1 + num2 + carry
        node = Node(sum % 10)
        node.next = pre
        pre = node
        carry = sum // 10
    if carry == 1:
        node = Node(1)
        node.next = pre
        pre = node
    return pre


def addList2(head1, head2):
    if head1 == None or head2 == None:
        raise Exception("Input Error!")
    head1 = reverseList(head1)
    head2 = reverseList(head2)
    pre1 = head1
    pre2 = head2
    pre = None
    carry = 0
    while pre1 != None or pre2 != None:
        sum = pre1.val + pre2.val + carry
        node = Node(sum % 10)
        node.next = pre
        pre = node
        carry = sum // 10
        pre1 = pre1.next
        pre2 = pre2.next
    if carry == 1:
        node = Node(1)
        node.next = pre
        pre = node
    reverseList(head1)
    reverseList(head2)
    return pre


#两个单链表相交的一系列问题
def getLoopNode(head):
    if head == None or head.next == None or head.next.next == None:
        return None
    slow = head.next
    fast = head.next.next
    while slow != fast:
        if fast.next == None or fast.next.next == None:
            return None
        slow = slow.next
        fast = fast.next.next
    fast = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    return slow


def noLoop(head1, head2):
    if head1 == None or head2 == None:
        return None
    cur1 = head1
    cur2 = head2
    n = 0
    while cur1.next != None:
        n += 1
        cur1 = cur1.next
    while cur2.next != None :
        n -= 1
        cur2 = cur2.next
    if cur1 != cur2:
        return None 
    cur1 = head1 if n >= 0 else head2
    cur2 = head1 if cur1 == head2 else head2
    n = abs(n)
    while n != 0:
        cur1 = cur1.next 
        n -= 1
    while cur1 != cur2:
        cur1 = cur1.next
        cur2 = cur2.next
    return cur1


def bothLoop(head1, node1, head2, node2):
    if head1 == None or head2 == None:
        return None
    if node1 == node2:
        cur1 = head1
        cur2 = head2
        n = 0
        while cur1 != node1:
            n += 1
            cur1 = cur1.next
        while cur2 != node1:
            n -= 1
            cur2 = cur2.next
        cur1 = head1 if n >= 0 else head2
        cur2 = head1 if cur1 == head2 else head2
        n = abs(n)
        while cur1 != 0:
            n -= 1
            cur1 = cur1.next
        while cur1 != cur2:
            cur1 = cur1.next
            cur2 = cur2.next
        return cur1
    else:
        cur1 = node1.next
        while cur1 != node1:
            if cur1 == node2:
                return node1
            cur1 = cur1.next
        return None


def getIntersectNode(head1, head2):
    if head1 == None or head2 == None:
        return None
    node1 = getLoopNode(head1)
    node2 = getLoopNode(head2)
    if node1 == None and node2 == None:
        return noLoop(head1, head2)
    if node1 != None and node2 != None:
        return bothNode(head1, node1, head2, node2)
    return None


#将单链表的每Ｋ个节点之间逆序
def reverseKNode(head, k):
    def reverse(stack, pre, next):
        while stack:
            cur = stack.pop()
            if pre != None:
                pre.next = cur
            pre = cur
        pre.next = next
        return pre


    if head == None or head.next == None or k < 2:
        return head
    stack = []
    cur = head
    newHead = head
    pre = None
    while cur != None:
        next = cur.next
        stack.append(cur)
        if len(stack) == k:
            pre = reverse(stack, pre, next)
            newHead = cur if newHead == head else newHead
        cur = next
    return newHead


def reverseKNode2(head, k):
    def reverse2(head, left, right):
        pre = None
        start = head
        while head != right:
            next = head.next
            head.next = pre
            pre = head
            head = next
        if left != None:
            left.next = pre
        start.next = right

    if head == None or head.next == None or k < 2:
        return head
    pre = None
    cur = head
    count = 0
    while cur != None:
        count += 1
        next = cur.next
        if count == k:
            start = head if pre == None else pre.next
            head = cur if pre == None else head
            reverse2(start, pre, next)
            pre = start
            count = 0
        cur = next
    return head


#删除无序单链表中值重复出现的节点
def removeRepeatNode(head):
    if head == None or head.next == None:
        return head
    hashSet = set()
    pre = head
    cur = head.next
    hashSet.add(head.val)
    while cur != None:
        next = cur.next
        if cur.val not in hashSet:
            hashSet.add(cur.val)
            pre = cur
        else:
            pre.next = next
        cur = next


def removeRepeatNode2(head):
    if head == None or head.next == None:
        return head
    while head != None:
        pre = head
        cur = head.next
        while cur != None:
            if cur.val == head.val:
                pre.next = cur.next
            else:
                pre = cur
            cur = cur.next
        head = head.next


#在单链表中删除指定值的节点
def removeValue1(head, num):
    if head == None:
        return None
    stack = []
    while head != None:
        if head.val != num:
            stack.append(head)
        head = head.next
    while stack:
        stack[-1].next = head
        head = stack.pop()
    return head


def removeValue2(head, num):
    if head == None:
        return head
    while head != None and head.val == num:
        head = head.next
    pre = head
    cur = head
    while cur != None:
        if cur.val == num:
            pre.next = cur.next
        else:
            pre = cur
        cur = cur.next


#将搜索二叉树转换成双向链表(未调试)
def convert1(head):
    def inOrderToQueue(head, queue):
        if head == None:
            return 
        inOrderToQueue(head.left, queue)
        queue.append(head)
        inOrderToQueue(head.right, queue)

    if head == None:
        return None
    queue = []
    inOrderToQueue(head, queue)
    head = queue.pop(0)
    head.left = None
    newHead = head
    while queue:
        node = queue.pop(0)
        head.right = node
        node.left = head
        head = node
    head.right = None
    return newHead


def convert2(head):
    def process(head):
        if head == None:
            return 
        leftE = process(head.left)
        rightE = process(head.right)
        leftH = None if leftE == None else leftE.right
        rightH = None if rightE == None else rightE.right
        if leftE != None and rightE != None:
            leftE.right = head
            head.left = leftE
            rightH.left = head
            head.right = rightH
            rightE.right = leftH
            return rightE
        elif leftE != None:
            leftE.right = head
            head.left = leftE
            head.right = leftH
            return head
        elif rightE != None:
            head.right = rightH
            rightH.left = head
            rightE.right = head
            return rightE
        else:
            head.right = head
            return head

    if head == None:
        return None
    tail = process(head)
    head = tail.right
    tail.right = None
    return head


#单链表的选择排序
def selectionSort(head):
    def getSmallestPre(head):
        if head == None:
            return None
        pre = head    #important
        smallest = head
        smallPre = None
        head = head.next
        while head != None:
            if head.val < smallest.val:
                smallest = head
                smallPre = pre
            pre = head
            head = head.next
        return smallPre
            

    if head == None or head.next == None:
        return head
    tail = None     #排序后链表的尾节点
    newHead = None
    cur = head
    small = None
    while cur != None:
        smallPre = getSmallestPre(cur)    #最小节点的前一个节点，方便删除节点
        if smallPre != None:
            small = smallPre.next
            smallPre.next = small.next
        else:
            small = cur
            cur = cur.next
        if tail == None:
            tail = small
            newHead = tail
        else:
            tail.next = small
            tail = small
    return newHead


#一种怪异的节点删除方式
def removeNode(node):
    if node == None:
        return None
    next = node.next
    if next == None:
        raise Exception("Can not remove last node.")
    node.val = next.val
    node.next = next.next
        

#向有环的环形链表中插入新节点
def insertNum(head, num):
    node = Node(num)
    if head == None:
        node.next = node
        return node
    pre = head
    cur = head.next
    while cur != head:
        if pre.val <= num and cur.val >= num:
            break
        pre = pre.next
        cur = cur.next
    pre.next = node
    node.next = cur
    return head if head.val < num else node


#合并两个有序的单链表
def mergeTwoLinks(head1, head2):
    if head1 == None or head2 == None:
        return head1 if head2 == None else head2
    head = head1 if head1.val < head2.val else head2
    cur1 = head1 if head == head1 else head2
    cur2 = head1 if head == head2 else head2
    pre = None
    while cur1 != None and cur2 != None:
        if cur1.val <= cur2.val:
            pre = cur1
            cur1 = cur1.next
        else:
            next = cur2.next
            pre.next = cur2
            cur2.next = cur1
            pre = cur2
            cur2 = next
    pre.next = cur1 if cur2 == None else cur2
    return head


#按照左右半区的方式重新组合单链表
def reCombination(head):
    if head == None or head.next == None:
        return head
    mid = head
    right = head.next
    while right.next != None and right.next.next != None:
        mid = mid.next
        right = right.next.next
    right = mid.next
    mid.next = None
    cur = head
    while cur.next != None:
        rightNext = right.next
        right.next = cur.next
        cur.next = right
        cur = right.next
        right = rightNext
    cur.next = right
    return head





arr1 = [1,3,4,5,7]
arr2 = [3,4,5,6,8]
test1 = Node()
t1 = test1
test2 = Node()
t2 = test2
for i in range(len(arr1)):
    test1.val = arr1[i]
    test2.val = arr2[i]
    if i < len(arr1)-1:
        test1.next = Node()
        test2.next = Node()
        test1 = test1.next
        test2 = test2.next
#test1.next = t1
#printCommonPart(t1, t2)
#head = removeLastKthNode1(t1, 2)
#head = removeMidNode(t1)
#head = removeByRatio(t1, 1, 4)
#head = reverseList(t1)
#head = reversePart(t1, 1, 3)
#head = josephusKill1(t1, 2)
#while head != None:
#    print(head.val, end=' ')
#    head = head.next
arr3 = [1,2,3,4,5,6,5,4,3,2,1]
test3 = Node()
t3 = test3
test4 = t3
for i in range(len(arr3)):
    t3.val = arr3[i]
    if i != len(arr3)-1:
        t3.next = Node()
        t3 = t3.next
#print(isPalindrome3(test3))
#listPartition2(test3, 4)
#while test3 != None:
#    print(test3.val, end=' ')
#    test3 = test3.next
#print()
#nodeArr = []
#for i in range(len(arr3)):
#    nodeArr.append(RandNode(arr3[i]))
#for i in range(len(nodeArr)-2):
#    nodeArr[i].next = nodeArr[i+1]
#    nodeArr[i].rand = nodeArr[i+2]
#nodeArr[-2].next = nodeArr[-1]
#cur = nodeArr[0]
#copyNode = copyListWithRand2(cur)
#copyNode1 = copyNode
#while cur != None:
#    print(cur.val, end=' ')
#    cur = cur.next
#print()
#cur = nodeArr[0]
#while cur != None:
#    print(cur.val, end=' ')
#    cur = cur.rand
#print()
#while copyNode != None:
#    print(copyNode.val, end=' ')
#    copyNode = copyNode.next
#print()
#while copyNode1 != None:
#    print(copyNode1.val, end=' ')
#    copyNode1 = copyNode1.rand
#print()
#head = addList2(t1, t2)
#while head != None:
#    print(head.val, end=' ')
#    head = head.next
#print()
#print(getIntersectNode(t1, t2))
tmp = test4
while tmp != None:
    print(tmp.val, end=' ')
    tmp = tmp.next
print()
#removeValue2(test4, 3)
#test4 = selectionSort(test4)
#removeNode(tmp)
#insertNum(test4, 3)
#t1 = mergeTwoLinks(t1, t2)
t1 = reCombination(test4)
while t1 != None:
    print(t1.val, end=' ')
    t1 = t1.next
print()
