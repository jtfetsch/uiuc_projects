#!/usr/bin/python2.7
"""
mp2.py

MP2 in python
"""

import argparse
import heapq
import itertools
import math
import os
import string
import cube

class PriorityQueue:
    """ Convenience class for insulating heapq calls to make priority queue """
    def __init__(self):
        self._heap = []
        self._present = set()

    # Adds the node to the top of the heap
    def add(self, state):
        """ Adds a new node to pqueue. If key already exists, then it keeps the node with lowest priority """
        chosen = (state.score, state)
        if state in self._present: # Replacement logic
            nodesToReinsert = []
            # Find existing and compare
            while self._heap:
                node = heapq.heappop(self._heap)
                (pri, val) = node
                if val == state:
                    cmpVal = cmp(val, state)
                    if cmpVal < 0:
                        chosen = node
                    elif cmpVal == 0:
                        nodesToReinsert.append(node)
                    nodesToReinsert.append(chosen)
                    break
                nodesToReinsert.append(node)
            # Reinsert
            for n in nodesToReinsert:
                heapq.heappush(self._heap, n)
            return 1
        else:
            # Add new node
            heapq.heappush(self._heap, chosen)
            self._present.add(state)
            return 0

    # Removes the node at the top of the heap
    def pop(self):
        (priority, result) = heapq.heappop(self._heap)
        self._present.discard(result)
        return result
    
    def clear(self):
        self._heap = []

    def __len__(self):
        return len(self._heap)
    
    def __str__(self):
        return str(self._heap)

    __repr__ = __str__


#
#  Parse command line arguments for filename and algorithms to run
#           
def parseArgs():
    parser = argparse.ArgumentParser(prog='MP2')
    parser.add_argument('-f', '--state_file', help='Cube file to load', required=True)
    parser.add_argument('-r', '--rotation', help='Solve with rotational invariance')
    return parser.parse_args()


'''
================================================================================
'''
def main():
    # Parse command line arguments
    args = parseArgs()
    fileName = args.state_file

    # Initialize the cube
    currentState = cube.Cube(generate=True)

    # Open the file and parse color positions
    currentState.load(fileName)
    print 'Solving 2x2 Rubik\'s cube starting in state:'

    # Display formatted output of initial cube state to console
    currentState.show()

    openList = PriorityQueue()
    closedList = set()
    copyState = cube.Cube(generate=True)
    currentState.copyTo(copyState)
    openList.add(copyState)
    finalResult = cube.Cube(generate=True)
    expandedCount = 0
    i = 0

    # Begin solving the cube
    while openList:
       currentState = openList.pop()
       if i % 1000 == 0:
           print currentState.path
           currentState.show()
       i = i + 1
       if currentState.isSolutionOne():
           currentState.copyTo(finalResult)
           break
       for successor in currentState.Successors():
          successor.updateScore()
          if successor not in closedList: # modify me to allow for inadmissible heuristic
             openList.add(successor)
       closedList.add(currentState)
       expandedCount = expandedCount + 1
    
    # Display final solution to console
    for each in finalResult.path:
       print each
    finalResult.show()
    print "Nodes Expanded: {}".format(expandedCount)
 

if __name__ == "__main__":
    main()
