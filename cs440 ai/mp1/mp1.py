#!/usr/bin/python2.7
"""
mp1.py

MP1 in python
"""

import argparse
import heapq
import itertools
import math
import os
import string
import csv

from collections import OrderedDict

def manhattan((x, y), (xp, yp)):
    return abs(x - xp) + abs(y - yp)

def euclidean((x, y), (xp, yp)):
    return math.sqrt((x - xp)**2 + (y - yp)**2)

def chebyshev((x, y), (xp, yp)):
    return max(abs(x - xp), abs(y - yp))

class MazeSolution:
    goalSymbols = "0123456789" + string.ascii_lowercase + string.ascii_uppercase

    """ Simple structure describing a maze solution """
    def __init__(self, maze, finalPath, expandedCount):
        self.maze = maze
        # Only care about coords at this point
        self.finalPath = map(lambda x: x.position, list(finalPath))
        self.expandedCount = expandedCount
        self.pathLen = len(self.finalPath)

    def writeOutput(self, fileName):
        """ Write the solution text file """
        solvedSymbols = self.__getSymbols()
        print("Writing solution to: '{}'".format(fileName))
        self.__createDirIfNotExists(fileName)
        with open(fileName, 'w') as f:
            for line in solvedSymbols:
                f.write(line.strip() + '\n')
        print("Done!")

    def __createDirIfNotExists(self, fileName):
        fileDir = os.path.dirname(fileName)
        if not os.path.exists(fileDir):
            os.makedirs(fileDir)

    def __getSymbols(self):
        """ Method to return the solved maze as a list """
        maze = self.maze
        (dimX, dimY) = maze.dimensions
        self.orderedGoals = OrderedDict()
        for x in self.finalPath:
            if x in maze.goals and x not in self.orderedGoals:
                self.orderedGoals[x] = True
        symbols = []
        for y in range(dimY):
            row = ""
            for x in range(dimX):
                symbol = self.__selectSymbol((x, y))
                row = row + symbol
            symbols.append(row)
        return symbols

    def __selectSymbol(self, pos):
        """ Determine which symbol to use based on coordinate """
        maze = self.maze
        orderedGoals = self.orderedGoals
        symbolMapSize = len(MazeSolution.goalSymbols)
        if maze.isWall(*pos):
            return '%'
        if pos == maze.start:
            return 'P'
        if pos in orderedGoals:
            if len(orderedGoals) == 1:
                return '.'
            idx = self.__findIndex(pos)
            if idx >= symbolMapSize:
                print("WARNING: Wrapping around goal symbol map since idx '{}' is greater than symbol map size '{}'".format(idx, symbolMapSize))
                idx = idx % symbolMapSize 
            return MazeSolution.goalSymbols[idx]
        if pos in self.finalPath:
            return '.'
        return ' '
    
    def __findIndex(self, pos):
        i = 0
        for x in self.orderedGoals:
            if x == pos:
                return i
            i = i + 1
        # Error occurred
        return -1

    def __len__(self):
        """ Length of solution is the length of final path """
        return len(self.finalPath)

class MazeState:
    """ Object representing maze state """
    def __init__(self, score, path, position, pathLen):
        self.score = score
        self.path = path
        self.position = position
        self.pathLen = pathLen

    def __str__(self):
        return "MazeState { " + str((self.score, self.position)) + " }"

    def __hash__(self):
        return hash(self.position)

    def __cmp__(self, other):
        return cmp(self.score, other.score)

    def __eq__(self, other):
        return self.position == other.position
    
    __repr__ = __str__

class PriorityQueue:
    """ Convenience class for insulating heapq calls to make priority queue """
    def __init__(self):
        self._heap = []
        self._present = set()

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
        else:
            # Add new node
            heapq.heappush(self._heap, chosen)
            self._present.add(state)

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

class Stack:
    """ Convenice class for stack (LIFO) """
    def __init__(self):
        self._stack = []

    def add(self, state):
        self._stack.append(state)

    def pop(self):
        return self._stack.pop()

    def clear(self):
        self._stack = []

    def __len__(self):
        return len(self._stack)

    def __str__(self):
        return str(self._stack)

    __repr__ = __str__

class Queue:
    """ Convenice class for queue (FIFO) """
    def __init__(self):
        self._queue = []

    def add(self, state):
        self._queue.append(state)

    def pop(self):
        return self._queue.pop(0)

    def clear(self):
        self._queue = []

    def __len__(self):
        return len(self._queue)

    def __str__(self):
        return str(self._queue)

    __repr__ = __str__

class Maze:
    """ Simple structure to describe loaded maze """
    def __init__(self, fileName=None, precompute=True, mazeCopy=None):
        walls = []
        start = (-1, -1)
        goals = set()
        maxX = -1
        maxY = -1
        row = 0
        if not mazeCopy:
            with open(fileName, 'r') as f:
                for line in f:
                    col = 0
                    walls.append([True if x == '%' else False for x in line])
                    for c in line:
                        if c == '\n':
                            continue # Skip newline character (don't count as column)
                        if c == 'P':
                            start = (col, row)
                        elif c == '.':
                            goals.add((col, row))
                        maxX = max(maxX, col)
                        col = col + 1
                    maxY = row
                    row = row + 1
        else:
            (maze, start, goal) = mazeCopy
            walls = maze.walls
            goals = set(goal)
            maxX = maze.dimX - 1
            maxY = maze.dimY - 1
        self.walls = walls
        self.start = start
        self.goals = goals
        self.dimensions = (maxX + 1, maxY + 1)
        (self.dimX, self.dimY) = self.dimensions
        self.filename = fileName
        self.hLookup = self._precompute() if precompute else None

    def isInBounds(self, x, y):
        """ Determine whether or not coordinates are in bounds """
        return x >= 0 and y >= 0 and y < len(self.walls) and x < len(self.walls[y])

    def isWall(self, x, y):
        """ Check if a coordinate is a wall """
        return self.isInBounds(x, y) and self.walls[y][x]
    
    def solve(self, (solver, initFrontier)):
        """ General method for solving the maze with a provided solver """
        start = self.start
        remGoals = self.goals
        expandedCount = 0
        frontier = initFrontier()
        frontier.add(MazeState(0, [], start, 0))
        closed = set()
        visited = {}
        closedSets = {}
        i = 1
        while frontier:
            #(pri, (curPath, curPos, curPathLen)) = heapq.heappop(toVisit)
            curNode = frontier.pop()

            newPath = list(curNode.path)
            newPath.append(curNode)

            # Remaining goals
            goals = remGoals.difference(map(lambda x: x.position, newPath))
            goalKey = str(sorted(list(goals)))

            # Closed sets are dependent on our final goal state
            if goalKey not in closedSets:
                closedSets[goalKey] = set()
            closed = closedSets[goalKey]

            # Check if we've already been here when searching for this specific set of goals
            if curNode in closed:
                continue

            # Keep track of algo status
            if i % 10000 == 0:
                print("Number of goals remaining: {}".format(len(goals)))
                i = 0
            i = i + 1

            # If we are on a goal but have more to find, clear our closed set and frontier
            isNewGoal = curNode.position in remGoals.difference(map(lambda x: x.position, curNode.path))
            if isNewGoal:
                closed = set()

            # If we have found a path to each goal, then we're done
            if not goals:
                return MazeSolution(self, newPath, expandedCount)

            # Otherwise, expand this node
            expandedPos = self.__expand(*curNode.position)
            newNodes = [x for x in zip([newPath] * 4, expandedPos, [curNode.pathLen + 1] * 4)] # Due to problem constraint, never more than 4 new positions
            expandedCount = expandedCount + 1

            # Mark node as visited
            closed.add(curNode)

            # Send the set of nodes to our solver
            newStates = solver(newNodes, goals)

            # Add updated nodes to the frontier
            for state in newStates:
                if state not in closed:
                    frontier.add(state)

        # Could not find a valid path
        return None

    def __expand(self, x, y):
        candidates = [(x, y + 1), (x - 1, y), (x, y - 1), (x + 1, y)]
        return filter(lambda x: self.isInBounds(*x) and not self.isWall(*x), candidates)

    def _precompute(self):
        allPos = [(x, y) for x in range(self.dimX) for y in range(self.dimY)]
        solveBFS = lambda x,y: Maze(precompute=False, mazeCopy  = (self, x, y)).solve((self.bfsSolver, Stack)) 
        dist = lambda x,y: manhattan(x, y)
        return dict((((start, goal), dist(start, goal)) for start in allPos for goal in self.goals))

    def bfsSolver(self, newNodes, goals):
        return [MazeState(-1, *x) for x in newNodes]
    
    def dfsSolver(self, newNodes, goals):
        return [MazeState(-1, *x) for x in newNodes]
    
    def gbfsSolver(self, newNodes, goals):
        # For using the gbfs solver, we always have only a single goal, so always just take index 0
        goal = next(iter(goals))
        h = lambda x: manhattan(x, goal)
        priFn = lambda x: h(x[1])
        return [MazeState(priFn(x), *x) for x in newNodes]
    
    def astarSolver(self, h, newNodes, goals):
        g = lambda x: x[2]
        priFn = lambda x: g(x) + h(x[1], goals)
        pathMax = lambda x: max(x[0][-1].score, priFn(x))
        return [MazeState(priFn(x), *x) for x in newNodes]

    def manhattanHeuristic(self, pos, goals):
        return max(map(lambda x: manhattan(pos, x), goals))

    def heuristic(self, pos, goals):
        """ Heuristic function
            Combination of 2 different heuristics
        """
        numGoalsRem = len(goals.difference(set(pos)))
        # h0 = If can move anywhere in single step, unrestricted (relax all movement constraint)
        h0 = numGoalsRem 
        # h1 = If we can move ignoring walls
        h1 = max(map(lambda x: self.hLookup[(pos, x)], goals))
        return max(h0, h1)
    
def main():
    args = parseArgs()
    fileName = args.maze_file
    maze = Maze(fileName)
    print("Read file '{}'. Found maze with dimensions: {} x {}".format(fileName, maze.dimensions[0], maze.dimensions[1]))
    print("Start position: ({}, {})".format(maze.start[0] + 1, maze.start[1] + 1))
    print("Goal positions: {}".format(reduce(lambda x,y: x + y, map(lambda x: "({}, {}) ".format(x[0] + 1, x[1] + 1), maze.goals))))
    
    print("Solving maze for select algorithms...")
    algorithms = {
                'BFS': (maze.bfsSolver, lambda: Queue()),
                'DFS': (maze.dfsSolver, lambda: Stack()),
                'GBFS': (maze.gbfsSolver, lambda: PriorityQueue()),
                'ASTAR': (lambda x,y: maze.astarSolver(maze.heuristic, x, y), lambda: PriorityQueue()),
                'ASTARM': (lambda x,y: maze.astarSolver(maze.manhattanHeuristic, x, y), lambda: PriorityQueue())
                }
    selectedAlgos = ['BFS', 'DFS', 'GBFS', 'ASTAR', 'ASTARM'] if not args.algorithms else args.algorithms

    fileNameNoExt = os.path.basename(fileName).split('.')[0]

    stats = []
    statsFile = 'solutions/{0}/{0}-stats.csv'.format(fileNameNoExt)
    for algo in selectedAlgos:
        solver = algorithms[algo] 
        solution = maze.solve(solver)
        print('{} results\n----------------------------'.format(algo))
        print("Path: {}".format(len(solution)))
        print("Expanded: {}".format(solution.expandedCount))
        print("Approx. Branching factor: {}".format(float(solution.expandedCount) ** (1.0 / float(len(solution.finalPath)))))

        fileNameBase = 'solutions/{0}/{0}-{1}'.format(fileNameNoExt, algo)
        mazeFileName = fileNameBase + '.txt'
        solution.writeOutput(mazeFileName)

        stats.append([algo, len(solution), solution.expandedCount])

        print('')

        print('Writing stats solution to: {}'.format(statsFile))
        writeStatsCsv(statsFile, stats)

def writeStatsCsv(filename, rows):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['algorithm', 'pathCost', 'expanded'])
        for row in rows:
            writer.writerow(row)

def parseArgs():
    algos = ["BFS", "DFS", "GBFS", "ASTAR", "ASTARM"]
    parser = argparse.ArgumentParser(prog='MP1')
    parser.add_argument('-f', '--maze_file', help='Maze file to load', required=True)
    parser.add_argument('-a', '--algorithms', nargs='+', help='algorithms to run (if none specified, all are run)', choices=algos)
    return parser.parse_args()

if __name__ == "__main__":
    main()
