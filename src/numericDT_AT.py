#!/usr/bin/python
"""
@authors: Michael Baldwin, Josh Engelsma, Adam Terwilliger
"""

import sys

def main(argv):
    if len(argv) < 2:
        print(usage())
    dataFilePath = argv[1]
    with open(dataFilePath, "r") as fh:
    	for line in fh:
        #print fh.readline().strip()
        	print line

def usage():
    return """
            python decision_tree.py [dataFile]
                [dataFile] - the path to the file that will be used to build the tree
            """

if __name__ == "__main__":
    main(sys.argv)