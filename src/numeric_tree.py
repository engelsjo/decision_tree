"""
@authors: Michael Baldwin, Josh Engelsma, Adam Terwilliger
"""

import sys
import math
import json

class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

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
        # a list of tuples
        self.numericValues = []
        self.isNumeric = False

    def __str__(self):
        return "Attribute Name: {} || AttributeValue(s): {}\n".format(self.attrName, self.attrValues)

    def __repr__(self):
        return str(self)

class BreakPoint(object):
    def __init__(self, avg, name1, name2):
        self.avg = avg
        self.name1 = name1
        self.name2 = name2

    def __str__(self):
        return "AVG: {} Name1: {} Name2: {}".format(self.avg, self.name1, self.name2)

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

class DecisionTree(object):
    def __init__(self, dataFilePath, targetNames=None, attributesAndValues=None, trainingExamples=None):
        self.targetNames = targetNames
        self.attributesAndValues = attributesAndValues
        self.trainingExamples = trainingExamples
        if not targetNames or not attributesAndValues or not trainingExamples:
            self.readFile(dataFilePath)
        self.rootNode = self.buildSubTree(self.trainingExamples, None)
        self.vizBetterNode = self.buildTree(self.trainingExamples, {'name' : 'Root', 'parent' : None})

    def pretty_print(self):
        # use a BFS algorithm to print out level order
        q = Queue();
        q.enqueue(self.rootNode)
        while (q.size() > 0):
            node = q.dequeue();
            print(node)
            for childrenKey in node.childrenNodes.keys():
                childNode = node.childrenNodes[childrenKey]
                if not childNode.isLeaf:
                    q.enqueue(childNode)

    def buildTree(self, trainingExamples, currVizNode):
        """
        method to build our tree with better labeling
        """
        divisionAttribute = self.getDivisionAttribute(trainingExamples)
        # divide up our data based on the attribute we got back
        subLists = {}
        for attrValue in divisionAttribute.attrValues:
            subLists[attrValue] = []
        for example in trainingExamples:
            # if the example attribute matches our division attributes, add training example to correct sublist
            for attribute in example.attributes:
                if attribute.attrName == divisionAttribute.attrName:
                    subLists[attribute.attrValues[0]].append(example)
        # assign all children for the current node
        if 'children' not in currVizNode:    
            currVizNode['children'] = []
        for val in divisionAttribute.attrValues:
            currVizNode['children'].append({'name' : "{} ~ {}".format(divisionAttribute.attrName, val), 'parent' : currVizNode['name'], 'children' : []})
        # recursively build for each sublist
        for subListKey in subLists:
            subList = subLists[subListKey]
            if subList == []: # no training examples, default to most common target value
                for child in currVizNode['children']:
                    if child['name'].split('~')[1].strip() == subListKey:
                        # create and append my leaf node
                        leaf = {'name' : 'Decision: {}'.format("e"), 'parent' : child['name']}
                        child['children'].append(leaf)
            elif not self.isLeafNode(subList):
                for child in currVizNode['children']:
                    if child['name'].split('~')[1].strip() == subListKey: # this is the child viz node to pass on
                        self.buildTree(subList, child)
                        break
            else:
                for child in currVizNode['children']:
                    if child['name'].split('~')[1].strip() == subListKey:
                        # create and append my leaf node
                        leaf = {'name' : 'Decision: {}'.format(subList[0].targetValue), 'parent' : child['name']}
                        child['children'].append(leaf)
        return currVizNode

    def buildSubTree(self, trainingExamples, currNode):
        """
        method to build our tree
        """
        if self.attributesAndValues == []:
            return currNode
        divisionAttribute = self.getDivisionAttribute(trainingExamples)
        if currNode == None: # rootptr
            currNode = TreeNode(divisionAttribute.attrName)
        else:
            currNode.name = divisionAttribute.attrName
        # divide up our data based on the attribute we got back
        subLists = {}
        for attrValue in divisionAttribute.attrValues:
            subLists[attrValue] = []
        for example in trainingExamples:
            # if the example attribute matches our division attributes, add training example to correct sublist
            for attribute in example.attributes:
                if attribute.attrName == divisionAttribute.attrName:
                    subLists[attribute.attrValues[0]].append(example)
        # check if any of the sublists would require us to return a leaf node
        for subListKey in subLists:
            childNode = TreeNode()
            subList = subLists[subListKey]
            if subList == []: # no training examples, default to most common target value
                childNode.isLeaf = True
                childNode.targetValue = "e"
                currNode.childrenNodes[subListKey] = childNode
            elif self.isLeafNode(subList):
                childNode.isLeaf = True
                childNode.targetValue = subList[0].targetValue
                currNode.childrenNodes[subListKey] = childNode
            else:
                currNode.childrenNodes[subListKey] = childNode
                # recursively build using each sublist
                self.buildSubTree(subList, childNode)
        #return the root node with everything built on
        return currNode

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
        for i, attribute in enumerate(self.attributesAndValues):
            # calculate the info gain for this attribute
            attributeValuesAndEntropy = self.calculateEntropy(dataSet, attribute)
            gainForAttribute = self.calculateGain(maxEntropy, len(dataSet), attributeValuesAndEntropy)
            if gainForAttribute >= currMaxGain:
                currMaxAttr = attribute
                currMaxGain = gainForAttribute
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
        elif attribute.isNumeric:
            return self.calculateNumericEntropy(dataSet, attribute)
        else:
            # this is not the first node, and we have multiple attribute values to consider
            for attribValue in attribute.attrValues:
                targetValueCounts = self.getTargetValueCounts(dataSet, attribute.attrName, attribValue)
                attributesValuesAndEntropy[attribValue] = self.calculateEntropyForumla(targetValueCounts)

        return attributesValuesAndEntropy 

    def calculateNumericEntropy(self, dataSet, attribute):
        breakPoints = []
        attributeIndexInDataPoint = None
        # build our list of tuples 
        valuesAndClasses = []
        for dataPoint in dataSet:
            valueIWant = None
            for i, attr in enumerate(dataPoint.attributes):
                if attr.attrName == attribute.attrName:
                    attributeIndexInDataPoint = i
                    valueIWant = attr.attrValues[0]
            valuesAndClasses.append((valueIWant, dataPoint.targetValue))
        # sort our list of tuples
        valuesAndClasses.sort()
        for i in range(len(valuesAndClasses) - 1):
            value1 = valuesAndClasses[i][0]
            value2 = valuesAndClasses[i+1][0]
            class1 = valuesAndClasses[i][1]
            class2 = valuesAndClasses[i+1][1]
            if value1 != value2 and class1 != class2:
                # we have a breakpoint
                avg = (float(value1) + float(value2)) / 2.0
                attrName1 = "<{}".format(avg)
                attrName2 = ">{}".format(avg)
                bp = BreakPoint(avg, attrName1, attrName2)
                breakPoints.append(bp)
        # we now have all breakpoints - calc entropy on breakpoints
        for breakPoint in breakPoints:
            # divide the data
            list1 = []
            list2 = []
            for dataPoint in dataSet:
                if float(dataPoint.attributes[attributeIndexInDataPoint].attrValues[0]) < breakPoint.avg:
                    list1.append(dataPoint)
                elif float(dataPoint.attributes[attributeIndexInDataPoint].attrValues[0]) > breakPoint.avg:
                    list2.append(dataPoint)
                else:
                    print("you messed something up")
            entropy1TargetValueCounts = self.getTargetValueCounts(list1, attribute.attrName, breakPoint.avg, (True, breakPoint.name1))
            entropy2TargetValueCounts = self.getTargetValueCounts(list2, attribute.attrName, breakPoint.avg, (True, breakPoint.name2))
            entropy1 = self.calculateEntropyForumla(entropy1TargetValueCounts)
            entropy2 = self.calculateEntropyForumla(entropy2TargetValueCounts)
            maxEntropy = self.calculateEntropy(dataSet, None)["initial"][1]
            gain = maxEntropy - float(entropy1[0])/len(dataSet) * float(entropy1[1]) - float(entropy2[0])/len(dataSet) * float(entropy2[1])
            print(gain)
        
    def getTargetValueCounts(self, dataSet, attributeName, attributeValue, isNumericData=(False, None)):
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
                if isNumericData[0] and "<" in str(isNumericData[1]):
                    # filter numeric data by less than
                    for attribute in example.attributes:
                        if attribute.attrName == attributeName:
                            if float(attribute.attrValues[0]) < float(attributeValue):
                                filteredData.append(example)
                elif isNumericData[0] and ">" in str(isNumericData[1]): 
                    # filter numeric data by greater than
                    for attribute in example.attributes:
                        if attribute.attrName == attributeName:
                            if float(attribute.attrValues[0]) > float(attributeValue):
                                filteredData.append(example)
                elif self.doesExampleMatchAttribVal(example, attributeName, attributeValue):
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

    def predictExamplePoint(self, examplePoint):
        """
        predicts a training example object, and returns a tuple
        where the 0th index of the tuple is the predicted value using our tree, and the
        1st index of the tuple is the actual value from the data point.
        """
        if self.rootNode == None:
            print("you must build a tree from training data before running predictions")
            sys.exit(1)
        currNode = self.rootNode
        while not currNode.isLeaf:
            currNodeAttrName = currNode.name
            # find the attribute in our example that matches the current Node attribute
            attributeOfInterest = None
            for attrib in examplePoint.attributes:
                if attrib.attrName == currNodeAttrName:
                    attributeOfInterest = attrib
                    break
            attributeOfInterestVal = attributeOfInterest.attrValues[0]
            currNode = currNode.childrenNodes[attributeOfInterestVal]
        return (currNode.targetValue, examplePoint.targetValue)

    def predictAllExamplesInFile(self, predictFile):
        """
        predicts all the training examples in a file, and returns a list of tuples
        where the 0th index of the tuple is the predicted value using our tree, and the
        1st index of the tuple is the actual value from the data point.
        """
        if self.rootNode == None:
            print("You must build a tree from training data before running predictions")
            sys.exit(1)
        values = []
        with open(predictFile, "r") as fh:
            for exampleLine in fh:
                if exampleLine[0] == ';':
                    continue #provides easy way to comment out data 
                exampleLineParts = exampleLine.split("D:")[1].strip().split()
                targetValue = exampleLineParts[-1]
                exampleAttributes = []
                for j in range(len(self.attributesAndValues)):
                    attrName = self.attributesAndValues[j].attrName
                    attrValues = [exampleLineParts[j]]
                    attribute = Attribute(attrName, attrValues)
                    exampleAttributes.append(attribute)
                te = TrainingExample(exampleAttributes, targetValue)
                predictVsActualTuple = self.predictExamplePoint(te)
                values.append(predictVsActualTuple)
        return values

    def predictedRate(self, listOfResults):
        """
        @listOfResults a list of tuples with actual vs predicted vals
        """
        nbrCorrect = 0
        for tup in listOfResults:
            predictVal = tup[0]
            actualVal = tup[1]
            if predictVal == actualVal:
                nbrCorrect += 1
        print("Correct: {}\nTotal: {}\nPercentage: {}%\n".format(nbrCorrect, len(listOfResults), float(nbrCorrect) / float(len(listOfResults))))

    def readFile(self, dataFilePath):
        targetNames = attributes = examples = None
        with open(dataFilePath, "r") as fh:
            nbrOfTargets = int(fh.readline().strip())
            targetValues = fh.readline().strip().split("T:")[1].split()
            nbrOfAttributes = int(fh.readline().strip())
            
            # build a list of attribute objects holding the attrName and attrValues
            attributes = []
            for attrNum in range(nbrOfAttributes):
                line = fh.readline().strip()
                attribute = Attribute(None, None)
                if "NA:" in line:
                    attrLine = line.split("NA:")[1]
                    attribute.isNumeric = True
                    attrLineParts = attrLine.split()
                    attrName = attrLineParts[0]
                    numericValues = []
                    for i in range(2, len(attrLineParts)):
                        numericValues.append(attrLineParts[i])
                    attribute.attrName = attrName
                    attribute.numericValues = numericValues
                else:
                    # A: is in line
                    attribute.isNumeric = False
                    attrLine = line.split("A:")[1]
                    attrLineParts = attrLine.split()
                    attrName = attrLineParts[0]
                    attrValues = []
                    for i in range(2, len(attrLineParts)):
                        attrValues.append(attrLineParts[i])
                    attribute.attrName = attrName
                    attribute.attrValues = attrValues
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
        self.targetNames = targetValues
        self.attributesAndValues = attributes
        self.trainingExamples = examples

def main(argv):
    if len(argv) < 2:
        print(usage())
    tree = DecisionTree(argv[1])
    # tests
    #results = tree.predictAllExamplesInFile(argv[2])
    #tree.predictedRate(results)
    # results = tree.predictAllExamplesInFile(argv[2])
    # tree.predictedRate(results)
    # export the dictionary to a json file
    with open('../viz/data/decision_tree.json', 'w') as jsonfile:
        json.dump(tree.vizBetterNode, jsonfile)
    

def usage():
    return """
            python decision_tree.py [dataFile]
                [dataFile] - the path to the file that will be used to build the tree
            """

if __name__ == "__main__":
    main(sys.argv)