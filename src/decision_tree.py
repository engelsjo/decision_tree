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
        return "Training Values: {}\nTargetValues: {}\n".format(self.attributes, self.targetValue)

    def __repr__(self):
        return self.__str__()

class Attribute(object):
    def __init__(self, attrName, attrValues):
        self.attrName = attrName
        self.attrValues = attrValues

    def __str__(self):
        return "Attribute Name: {}\nAttributeValues: {}\n".format(self.attrName, self.attrValues)

    def __repr__(self):
        return self.__str__()

class DecisionTree(object):
    def __init__(self, nbrOfTargets, targetNames, nbrOfAttributes, attributesAndValues, nbrOfTrainingExamples, trainingExamples):
        self.nbrOfTargets = nbrOfTargets
        self.targetNames = targetNames
        self.nbrOfAttributes = nbrOfAttributes
        self.attributesAndValues = attributesAndValues
        self.nbrOfTrainingExamples = nbrOfTrainingExamples
        self.trainingExamples = trainingExamples
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
    with open(dataFilePath, "r") as fh:
        nbrOfTargetsLine = fh.readline().strip()
        targetValuesLine = fh.readline().strip()
        nbrOfAttributes = int(fh.readline().strip())
        
        # build a list of attribute objects holding the attrName and attrValues
        attributes = [Attribute(None, None)] * nbrOfAttributes
        for attrNum in range(nbrOfAttributes):
            attrLine = fh.readline().strip().split("A:")[1]
            attrLineParts = attrLine.split()
            attrName = attrLineParts[0]
            attributes[attrNum].attrName = attrName
            attrValues = []
            for i in range(2, len(attrLineParts)):
                attrValues.append(attrLineParts[i])
            attributes[attrNum].attrValues = attrValues
                
        # build a list of all the example data
        nbrOfExamples = int(fh.readline().strip())
        examples = []
        for i in range(nbrOfExamples):
            exampleLine = fh.readline().strip().split("D:")[1]
            exampleLineParts = exampleLine.split()
            targetValue = exampleLineParts[-1]
            exampleAttributes = [Attribute(None, None)] * nbrOfAttributes




    #dt = DecisionTree()

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