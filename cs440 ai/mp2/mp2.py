#!/usr/bin/env python
"""
mp2.py
"""

import argparse, itertools, logging, sys, pickle

logging.basicConfig(level=logging.INFO)

EXPANDED_COUNT = 0
ASSIGNMENT_ORDER = []
SOLUTIONS = []

class State:
    def __init__(self, board, words):
        self.board = [[col for col in row] for row in board]
        self.remaining = set()
        self.vars = set(words)
        self.moveHistory = []
        self.adjacency = {}
        self.hintPositions = set()
        self.hintTypes = {}
        self.hintCount = {}
        for row in range(9):
            for col in range(9):
                if board[row][col] != '_':
                    self.hintPositions.add((row, col))
                    pos = (row, col)
                    val = board[row][col]
                    if pos not in self.hintTypes:
                        self.hintTypes[pos] = val
                    if val not in self.hintCount:
                        self.hintCount[val] = 0
                    self.hintCount[val] += 1
                self._calculateAdjacent((row, col))
                # Words are no shorter than 3 chars in our subsets
                for length in range(3, 10):
                    vVar = ('V', (row, col))
                    hVar = ('H', (row, col))
                    if self.isValueInBounds(vVar):
                        self.remaining.add(vVar)
                    if self.isValueInBounds(hVar):
                        self.remaining.add(hVar)
    
    def isValueInBounds(self, var):
        return True
        (dir, (row, col), length) = var
        if dir == 'V':
            return (row + length) <= 9
        return (col + length) <= 9
    
    def makeAssignment(self, word, value):
        (dir, (row, col)) = value
        length = len(word)
        consistent = True
        blankPositions = set()
        hintPos = None
        if dir == 'V':
            start = row
            idx = 0
            for cellRow in range(start, start + length):
                cell = self.board[cellRow][col]
                if word[idx] in self.getAdjacentValues((cellRow, col)).difference([cell]):
                    consistent = False
                    break
                if cell != '_' and cell != word[idx]:
                    consistent = False
                    break
                elif cell == '_':
                    blankPositions.add((cellRow, col))
                    self.board[cellRow][col] = word[idx]
                if (cellRow, col) in self.hintPositions:
                    hintPos = (cellRow, col)
                idx += 1
        else:
            start = col
            idx = 0
            for cellCol in range(start, start + length):
                cell = self.board[row][cellCol]
                if word[idx] in self.getAdjacentValues((row, cellCol)).difference([cell]):
                    consistent = False
                    break
                if cell != '_' and cell != word[idx]:
                    consistent = False
                    break
                elif cell == '_':
                    blankPositions.add((row, cellCol))
                    self.board[row][cellCol] = word[idx]
                if (row, cellCol) in self.hintPositions:
                    hintPos = (row, cellCol)
                idx += 1
        removedValues = blankPositions #self._fullyAssigned()
        if hintPos:
            self.hintPositions.discard(hintPos)
            self.hintCount[self.hintTypes[hintPos]] -= 1
        # TODO: This should be controlled by variable selection...
        #elif len(self.hintPositions) != 0 and 'T' in word:
        #    consistent = False
        self.vars = self.vars.difference([word])
        self.moveHistory.append((word, removedValues, blankPositions, hintPos))
        return consistent
    
    def undoAssignment(self):
        if len(self.moveHistory) == 0:
            return
        (word, removedVars, blankPositions, hintPos) = self.moveHistory.pop()
        if hintPos:
            self.hintPositions.add(hintPos)
            self.hintCount[self.hintTypes[hintPos]] += 1
        self.vars.add(word)
        #self.remaining = self.remaining.union(removedVars)
        for (row, col) in blankPositions:
            self.board[row][col] = '_'
    
    def printBoard(self):
        for row in self.board:
            for v in row:
                print v,
            print ""
    
    def getAdjacentValues(self, pos):
        values = set()
        # TODO: Could update this as we go too if performance is an issue
        for (row, col) in self.adjacency[pos]:
            values.add(self.board[row][col])
        return values.difference(['_'])

    def _calculateAdjacent(self, pos):
        (row, col) = pos
        adjacent = set()
        for row in range(9):
            for col in range(9):
                cell = (row, col)
                if State._cellsAreAdjacent(pos, cell):
                    adjacent.add(cell)
        self.adjacency[pos] = adjacent

    @staticmethod
    def _cellsAreAdjacent((vRow, vCol), (oRow, oCol)):
        # Rely on integer math
        vCell = (vRow / 3, vCol / 3)
        oCell = (oRow / 3, oCol / 3)
        return vRow == oRow or vCol == oCol or vCell == oCell

def wordFits(word, (dir, (row, col))):
    wordLen = len(word)
    if dir == 'V':
        return row + wordLen <= 9
    return col + wordLen <= 9

def mostConstrainedVar(state, (hintVal, hintCount)):
    """ Returns a list containing most constrained values.
    """
    possibilities = {}
    for word in state.vars:
        for value in state.remaining:
            if wordFits(word, value):
                if word not in possibilities:
                    possibilities[word] = set()
                possibilities[word].add(value)
    minVal = 10000 # Effectively inf
    bestVars = []
    for key in possibilities:
        value = len(possibilities[key]) if hintVal and hintVal not in key else -1
        #print key, value
        if value < minVal:
            minVal = value
            bestVars = [key]
        elif value == minVal:
            bestVars.append(key)
    if hintVal and minVal != -1:
        return [] # Could not find _any_ variables to fill the position with hint
    # There are imposter words... Need to try all viable alternatives :(
    return [x for x in possibilities]

def mostConstrainingVar(board, validVars, (hintVal, hintCount)):
    if len(validVars) == 0:
        return None # Need to handle this if this ever happens?
    sortedVars = sorted(validVars, key=len, reverse=True)
    if hintVal:
        return filter(lambda x: hintVal in x, sortedVars)[0]
    # With impostor vars need to consider alternatives :(
    return sortedVars[0]

def selectUnassigned(state, hint):
    return mostConstrainingVar(state, mostConstrainedVar(state, hint), hint)

def selectHint(state):
    if len(state.hintPositions) == 0:
        return (None, None)
    bestVal = None
    bestCount = None
    maxWord = -1
    for pos in state.hintPositions:
        posVal = state.hintTypes[pos]
        posCount = state.hintCount[posVal]
        thisMaxWordLen = max([len(x) for x in state.vars if posVal in x])
        if thisMaxWordLen > maxWord:#posCount < bestCount or not bestCount:
            bestVal = posVal
            bestCount = posCount
            maxWord = thisMaxWordLen
    return (bestVal, bestCount)

def orderDomainValues(word, state, (hintVal, hintCount)):
    validValues = [x for x in state.remaining if wordFits(word, x)]
    length = len(word)
    hintIdx = word.index(hintVal) if hintVal and hintVal in word else -1
    def numBlank((dir, (row, col))):
        count = 0
        start = row if dir == 'V' else col
        multiplier = 1
        if hintIdx != -1:
            if dir == 'V':
                multiplier = -1 if (row + hintIdx, col) in state.hintPositions else 1
            else:
                multiplier = -1 if (row, col + hintIdx) in state.hintPositions else 1
        for x in range(start, start + length):
            if dir == 'V':
                if state.board[x][col] == '_':
                    count += 1
            else:
                if state.board[row][x] == '_':
                    count += 1
        return count * multiplier
    values = map(lambda x: (numBlank(x), x), validValues)
    negative = [x for x in values if x[0] < 0]
    sortedValues = sorted(negative, key=lambda x:x[0], reverse=True)
    positive = [x for x in values if x[0] >= 0]
    sortedValues.extend(sorted(positive, key=lambda x:x[0]))
    return map(lambda x: x[1], sortedValues)

def isSolved(state):
    for row in state.board:
        for val in row:
            if val == '_':
                return False
    return True
    
def solveSudoku(state):
    global EXPANDED_COUNT, ASSIGNMENT_ORDER, SOLUTIONS
    hint = selectHint(state)
    (hintVal, hintCount) = hint
    # Complete assignment found
    if isSolved(state):
        return True
    elif len(state.vars) == 0 or (hintVal and len(filter(lambda x: hintVal in x, state.vars)) < hintCount):
        print "drat"
        return False
    # Expands a new node
    EXPANDED_COUNT += 1
    # Select a variable
    if "GRYPHON" in "".join([state.board[2][x] for x in range(9)]):
        state.printBoard()
    vars = selectUnassigned(state, hint)
    if len(vars) == 0:
        return False
    for var in vars:
        # Order possible values
        values = orderDomainValues(var, state, hint)
        if EXPANDED_COUNT % 1000 == 0:
            state.printBoard()
            print "----------"
        #print "Selecting: {} with {} values. First is: {}. {} vars with T remaining.".format(var, len(values), values[0], len(filter(lambda x: 'T' in x, state.vars)))
        for value in values:
            # If assignment not consistent, don't recurse
            if state.makeAssignment(var, value):
                ASSIGNMENT_ORDER.append((var, value))
                if solveSudoku(state):
                    # Save the solution and keep looking for _all_ solutions
                    print "Found a solution!"
                    SOLUTIONS.append((State(state.board, state.vars), list(ASSIGNMENT_ORDER)))
                    with open('state2.pickle', 'w') as f:
                        pickle.dump((EXPANDED_COUNT, SOLUTIONS), f)
                        print 'Saved state out to "state.pickle"'
                    return True
                ASSIGNMENT_ORDER.pop()
            state.undoAssignment()
    # state.vars.discard(var)
    # result = solveSudoku(state)
    # state.vars.add(var)
    return False

def main():
    args = parseArgs()
    grid = loadGrid(args.grid_file)
    wordBank = loadWords(args.wordbank_file)
    state = State(grid, wordBank)
    solveSudoku(state)
    state.printBoard()
    print "Assignment order:"
    print "-----------------"
    for (word, (dir, (row, col))) in ASSIGNMENT_ORDER:
        print "{}, {}, {}: {}".format(dir, row, col, word)
    print ""
    print "Expanded count: {}".format(EXPANDED_COUNT)

def loadWords(fName):
    """ Load words
      Represented as a simple list of words
    """
    with open(fName, 'r') as f:
        return [word.strip().upper() for word in f]

def loadGrid(fName):
    """ Load grid
     A grid is a row-major list of lists (i.e. each row is a list of characters)
     We should probably wrap this in a class for convenience, but we can treat
     each row and column as a set since the contained values should be unique
    
     NOTE: This particular reading uses a list rather than a set since we _do_
           need to retain positional information
    """
    with open(fName, 'r') as f:
        return [[letter for letter in line.strip()] for line in f]

def parseArgs():
    parser = argparse.ArgumentParser(prog='MP2')
    parser.add_argument('-g', '--grid_file', help='Grid file to load', required=True)
    parser.add_argument('-b', '--wordbank_file', help='Word bank file to load')
    return parser.parse_args()

if __name__ == "__main__":
    main()
