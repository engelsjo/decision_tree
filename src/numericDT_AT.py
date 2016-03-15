#!/usr/bin/python
"""
@authors: Michael Baldwin, Josh Engelsma, Adam Terwilliger
"""

import sys
import numpy as np

def main(argv):
    if len(argv) < 2:
        print(usage())
    dataFilePath = argv[1]
    #irisDict = {}    
    irisMatrix = np.zeros((150,4))
    with open(dataFilePath, "r") as fh:
    	i = 0
    	for line in fh:
        #print fh.readline().strip()
        	lineParts = line.strip("\n").split(",")
        	#irisDict[(lineParts[4], i)] = [float(lineParts[0]), float(lineParts[1]),
        	#							float(lineParts[2]), float(lineParts[3])]
        	
            
            for j in range(4):
        		irisMatrix[i,j] = float(lineParts[j])
        	i +=1
    
    a = irisMatrix
    b = a[a[:,0].argsort()]
    print irisMatrix[0]
    print b


def usage():
    return """
            python decision_tree.py [dataFile]
                [dataFile] - the path to the file that will be used to build the tree
            """

if __name__ == "__main__":
    main(sys.argv)