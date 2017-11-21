class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def createTree(root):
    key = input("Enter a key: ")
    if key == '#':
        root = None
    else:
        root = TreeNode(key)
        root.left = createTree(root.left)
        root.right = createTree(root.right)
    return root

def midOrder(root):
    if root:
        midOrder(root.left)
        print(root.val, end=' ')
        midOrder(root.right)

def postOrder(root):
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

def printEdge1(root):
    def getHeight(root, height=0):
        if not root:
            return 0
        return max(getHeight(root.left, height+1), getHeight(root.right, height+1)) + 1
    
    def getMap(root, i, map):
        if not root:
            return
        if map[i][0] == None:
            map[i][0] = root
        map[i][1] = root
        getMap(root.left, i+1, map)
        getMap(root.right, i+1, map)

    def printLeafNotInMap(root, i, map):
        if not root:
            return
        if not root.left and not root.right and root != map[i][0] and \
                root != map[i][1]:
            print(root.val, end=' ')
        printLeafNotInMap(root.left, i+1, map)
        printLeafNotInMap(root.right, i+1, map)

    if not root:
        return
    height = getHeight(root)
    map = [[None for i in range(2)] for j in range(height)]
    getMap(root, 0, map)
    for i in range(len(map)):
        print(map[i][0].val, end=' ')
    printLeafNotInMap(root, 0, map)
    for i in range(len(map)-1, -1, -1):
        if map[i][0] != map[i][1]:
            print(map[i][1].val, end=' ')

def printEdge2(root):
    def printLeft(root, isPrint):
        if not root:
            return
        if isPrint or (root.left == None and root.right == None):
            print(root.val, end=' ')
        printLeft(root.left, isPrint)
        printLeft(root.right, bool(isPrint and root.left == None))

    def printRight(root, isPrint):
        if not root:
            return
        printRight(root.left, bool(isPrint and root.right == None))
        printRight(root.right, isPrint)
        if isPrint or (root.left == None and root.right == None):
            print(root.val, end=' ')


    if not root:
        return
    print(root.val, end=' ')
    if root.left and root.right:
        printLeft(root.left, True)
        printRight(root.right, True)
    elif root.left:
        printEdge2(root.left)
    elif root.right:
        printEdge2(root.right)

t = None
t = createTree(t)
print("midOrder: ")
midOrder(t)
print()
print("postOrder: ")
postOrder(t)
print()
print("printEdgeNode: ")
printEdge2(t)

