class TreeNode(object):
    def __init__(self, name=None):
        self.name = name
        self.isLeaf = False
        self.targetValue = None
        self.childrenNodes = {}

    def __str__(self):
        if not self.isLeaf:
            return "NODE: {} - Branches: {} ".format(self.name, self.childrenNodes.keys())
        return "** LeafNode ** Target: {} ".format(self.targetValue)

    def __repr__(self):
        return str(self)