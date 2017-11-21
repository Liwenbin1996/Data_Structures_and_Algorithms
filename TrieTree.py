class TrieNode:
    def __init__(self):
        self.path = 0
        self.end = 0 
        self.map = [None for i in range(26)]

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        if word == None:
            return
        node = self.root
        for i in word:
            index = ord(i) - ord('a')
            if node.map[index] == None:
                node.map[index] = TrieNode()
            node = node.map[index]
            node.path += 1
        node.end += 1
        print("Inset successful")

    def search(self, word):
        if word == None:
            return False
        node = self.root
        for i in word:
            index = ord(i) - ord('a')
            if node.map[index] == None:
                return False
            node = node.map[index]
        return True if node.end != 0 else False

    def delete(self, word):
        if self.search(word):
            node = self.root
            for i in word:
                index = ord(i) - ord('a')
                node.map[index].path -= 1
                if node.map[index].path == 0:
                    node.map[index] = None
                    return
                node = node.map[index]
            node.end -= 1

    def prefixNumber(self, pre):
        if pre == None:
            return
        node = self.root
        for i in pre:
            index = ord(i) - ord('a')
            if node.map[index] == None:
                return 0
            node = node.map[index]
        return node.path



TrieTree = Trie()
TrieTree.insert("hello")
print(TrieTree.search("hello"))
print(TrieTree.prefixNumber("hel"))
TrieTree.delete("hello")
print(TrieTree.search("hello"))
TrieTree.delete("hello")

