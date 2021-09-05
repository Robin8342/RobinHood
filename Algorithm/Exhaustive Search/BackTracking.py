#완전탐색 아이디어에서 불필요한 분기(Bracnch)를 가지치기(Pruning) 한다.
#탐색을 하면서 정답이 될 수 없는 조건은 가지치기한다
#해를 찾기 위해 탐색할 필요가 있는 모든 후보들을 포함하는 트리.
#루트에서 출발하여 모든 노드를 방문함.

class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

    def __str__(self):
        return str(self.data)

class Tree:
    def __init__(self):
        self.root = None
    #전위 순회
    def BackTrackingPreorder(self, node):
        print(node, end='')
        if not node.left == None : self.BackTrackingPreorder(node.left)
        if not node.right == None : self.BackTrackingPreorder(node.left)

    #중위 순회
    def BackTrackingInorder(self, node):
        if not node.left == None : self.BackTrackingInorder(node.left)
        print(node, end='')
        if not node.right == None : self.BackTrackingInorder(node.right)

    #후위 순회
    def BackTrackingPostorder(self, node):
        if not node.left == None : self.BackTrackingPostorder(node.left)
        if not node.right == None: self.BackTrackingPostorder(node.right)
        print(node, end='')

    def BackTrackingRoot(self, node, left_node, right_node):
        if self.root ==None:
            self.root = node
        node.left = left_node
        node.right = right_node

node = []
node.append(Node('-'))
node.append(Node('*'))
node.append(Node('/'))
node.append(Node('A'))
node.append(Node('B'))
node.append(Node('C'))
node.append(Node('D'))

m_tree = Tree()

for i in range(int(len(node)/2)):
    m_tree.BackTrackingRoot(node[i],node[i*2+1],node[i*2+2])


print(m_tree.root)
print("#################")
print('전위' ,end='') ; m_tree.BackTrackingPreorder(m_tree.root)
print("#################")
print('전위' ,end='') ; m_tree.BackTrackingInorder(m_tree.root)
print("#################")
print('전위' ,end='') ; m_tree.BackTrackingPostorder(m_tree.root)
