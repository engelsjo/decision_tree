"""
@authors: Michael Baldwin, Josh Engelsma, Adam Terwilliger
"""

import sys

class TrainingExample(object):
    def __init__(self, attributesAndValues, targetValue):
        self.attributesAndValues = attributesAndValues
        self.targetValue = targetValue


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
    dt = DecisionTree()

def usage():
    """
    print out how to operate the program
    """
    



if __name__ == "__main__":
    main(sys.argv)