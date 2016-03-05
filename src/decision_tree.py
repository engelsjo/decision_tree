"""
@authors: Michael Baldwin, Josh Engelsma, Adam Terwilliger
"""

import sys
import math

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

class TreeNode(object):
    def __init__(self, name, d):
        self.name = name
        self.isLeaf = False
        self.targetValue = None

class DecisionTree(object):
    def __init__(self, targetNames, attributesAndValues, trainingExamples):
        self.targetNames = targetNames
        self.attributesAndValues = attributesAndValues
        self.trainingExamples = trainingExamples
        self.rootNode = self.buildTree(self.trainingExamples)
        print(self.rootNode)

    def buildTree(self, trainingExamples):
        """
        method to build our tree
        """
        divisionAttribute = self.getDivisionAttribute(trainingExamples)
        # divide up our data based on the attribute we got back
        subLists = {}
        for attrValue in divisionAttribute.attrValues:
            subLists[attrValue] = []
        for example in trainingExamples:
            # if the example attribute matches our division attributes, add training example to correct sublist
            for attribute in example.attributes:
                if attribute.attrName == divisionAttribute.attributeName:
                    subLists[attribute.attrValues[0]].append(example)
        # check if any of the sublists would require us to return a leaf node
        for subListKey in subLists:
            subList = subLists[subListKey]
            if self.isLeafNode(subList):
                # create a leaf Node
                node = TreeNode()
        #now I have 'subLists' which is a dictionary of attributes Values matched to dataSubSets
        for subListKey in subLists:
            self.buildTree(subLists[subListKey])  

    def isLeafNode(self, subList):
        """
        @param subList: a List of TrainingExample objects.
        """
        currTrainingValue = subList[0].targetValue
        for example in subList:
            if example.targetValue != currTrainingValue:
                return False
        return True

    def getDivisionAttribute(self, dataSet):
        maxEntropy = self.calculateEntropy(dataSet, None)["initial"][1]
        currMaxGain = 0
        currMaxAttr = None
        deletionIndex = None
        for i, attribute in enumerate(self.attributesAndValues):
            # calculate the info gain for this attribute
            attributeValuesAndEntropy = self.calculateEntropy(dataSet, attribute)
            gainForAttribute = self.calculateGain(maxEntropy, len(dataSet), attributeValuesAndEntropy)
            print(gainForAttribute)
            if gainForAttribute > currMaxGain:
                currMaxAttr = attribute
                currMaxGain = gainForAttribute
                deletionIndex = i
        # update the list of attributes we can divide on
        self.attributesAndValues.pop(deletionIndex)
        return currMaxAttr

    def calculateEntropy(self, dataSet, attribute):
        """
        @param exampleData: a list of TrainingExamples -> given the node we are at.
        @return 
        """
        attributesValuesAndEntropy = {}
        if not attribute:
            # this is the first node, so use all of the training examples
            targetValueCounts = self.getTargetValueCounts(dataSet, "initial", None)
            attributesValuesAndEntropy["initial"] = self.calculateEntropyForumla(targetValueCounts)
        else:
            # this is not the first node, and we have multiple attribute values to consider
            for attribValue in attribute.attrValues:
                targetValueCounts = self.getTargetValueCounts(dataSet, attribute.attrName, attribValue)
                attributesValuesAndEntropy[attribValue] = self.calculateEntropyForumla(targetValueCounts)
        return attributesValuesAndEntropy 
        
    def getTargetValueCounts(self, dataSet, attributeName, attributeValue):
        """
        returns a dictionary where they key is the target value, and the value is the count found 
        matching that target value given that the attributeValue matches.
        """
        targetValueCounts = {}
        if attributeName == "initial":
            for example in dataSet:
                if example.targetValue not in targetValueCounts:
                    targetValueCounts[example.targetValue] = 1
                else:
                    targetValueCounts[example.targetValue] += 1
            return targetValueCounts
        else:
            # first filter out data from examples which match our attributeValue
            filteredData = []
            for example in dataSet:
                # filter out data set to only use examples with attributeName matching our attributeValue
                if self.doesExampleMatchAttribVal(example, attributeName, attributeValue):
                    filteredData.append(example)
            # now using our filtered data get the target counts
            for example in filteredData:
                if example.targetValue not in targetValueCounts:
                    targetValueCounts[example.targetValue] = 1
                else:
                    targetValueCounts[example.targetValue] += 1
            return targetValueCounts

    def doesExampleMatchAttribVal(self, examplePoint, attributeName, attributeValue):
        for attribute in examplePoint.attributes:
            if attribute.attrName == attributeName and attribute.attrValues[0] == attributeValue:
                return True
        return False

    def calculateEntropyForumla(self, targetValueCounts):
        """
        Helper function used by 'calculateEntropyForAttribute'
        @param countForTarget: The number of examples matching the target value.
        """
        entropy = 0
        totalNbrMatchingAttributeValue = 0
        for targetValue in targetValueCounts:
            totalNbrMatchingAttributeValue += targetValueCounts[targetValue]
        for key in targetValueCounts:
            probability = float(targetValueCounts[key]) / float(totalNbrMatchingAttributeValue)
            entropy += probability * math.log(probability, 2)
        entropy = entropy * -1
        return (totalNbrMatchingAttributeValue,entropy)

    def calculateGain(self, maxEntropy, dataSetSize, attributeValuesAndEntropy):
        """
        @param attributeValuesAndEntropy: dictionary where they key is an attribute value, and the value
        is the entropy of that attribute value
        """
        gain = maxEntropy
        for key in attributeValuesAndEntropy:
            totalNbrMatchingAttributeValue = attributeValuesAndEntropy[key][0]
            entropyForAttrValue = attributeValuesAndEntropy[key][1]
            gain = gain - (float(totalNbrMatchingAttributeValue) / float(dataSetSize)) * entropyForAttrValue
        return gain

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