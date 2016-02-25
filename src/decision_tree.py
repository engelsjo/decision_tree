"""
@authors: Michael Baldwin, Josh Engelsma, Adam Terwilliger
"""

import sys

class TrainingExample(object):
    def __init__(self, attributes, targetValue):
        # self.attributes will be a list of Attribute objects where each attribute has a list of 1 value
        self.attributes = attributes
        self.targetValue = targetValue

    def __str__(self):
        return "\n***EXAMPLE DATA POINT***\nTraining Values:\n {}\nTargetValues: {}\n".format(self.attributes, self.targetValue)

    def __repr__(self):
        return str(self)

class Attribute(object):
    def __init__(self, attrName, attrValues):
        self.attrName = attrName
        self.attrValues = attrValues

    def __str__(self):
        return "Attribute Name: {} || AttributeValue(s): {}\n".format(self.attrName, self.attrValues)

    def __repr__(self):
        return str(self)

class DecisionTree(object):
    def __init__(self, targetNames, attributesAndValues, trainingExamples):
        self.targetNames = targetNames
        self.attributesAndValues = attributesAndValues
        self.trainingExamples = trainingExamples
        self.rootNode = None
        self.buildTree()

    def buildTree(self):
        """
        method to build our tree
        """
        self.rootNode = None

def main(argv):
    if len(argv) < 1:
        print(usage())
    dataFilePath = argv[1]
    targetNames = attributes = examples = None
    with open(dataFilePath, "r") as fh:
        nbrOfTargets = int(fh.readline().strip())
        targetValues = fh.readline().strip().split("T:")[1].split()
        nbrOfAttributes = int(fh.readline().strip())
        
        # build a list of attribute objects holding the attrName and attrValues
        attributes = []
        for attrNum in range(nbrOfAttributes):
            attrLine = fh.readline().strip().split("A:")[1]
            attrLineParts = attrLine.split()
            attrName = attrLineParts[0]
            attrValues = []
            for i in range(2, len(attrLineParts)):
                attrValues.append(attrLineParts[i])
            attribute = Attribute(attrName, attrValues)
            attributes.append(attribute)
        
        # build a list of all the example data
        nbrOfExamples = int(fh.readline().strip())
        examples = []
        for i in range(nbrOfExamples):
            exampleLine = fh.readline().strip().split("D:")[1]
            exampleLineParts = exampleLine.split()
            targetValue = exampleLineParts[-1]
            exampleAttributes = []
            for j in range(nbrOfAttributes):
                attrName = attributes[j].attrName
                attrValues = [exampleLineParts[j]]
                attribute = Attribute(attrName, attrValues)
                exampleAttributes.append(attribute)
            te = TrainingExample(exampleAttributes, targetValue)
            examples.append(te)

    tree = DecisionTree(targetValues, attributes, examples)


def usage():
    """
    return str of how to operate the program
    """
    return """
    python decision_tree.py [dataFile]
        [dataFile] - the path to the file that will be used to build the tree
    """




if __name__ == "__main__":
    main(sys.argv)