#!/usr/bin/env python2.7
"""
program for running minimax search for the breakthrough game.
"""

from BreakthroughBoard import BreakthroughBoard
from random import shuffle
import argparse, time
import sys

MAX_DEPTH = [3, 3]
EXPANDED_COUNT = [0, 0]

def offensiveEvalFn(board, currentPlayer):
    value = board.isGameOver()
    if value == 0:
        # Use weighted heuristics
        # 0 to (numRows-1) points for each piece closer to goal depending on distance
        # 2 points for every stolen piece
        otherPlayer = 1 if currentPlayer == 2 else 2
        positions = board.getPiecePositions()
        opponents = board.getOpponentPiecePositions()
        #distances = [dist(pos, opp) for pos in positions for opp in opponents]
        numThreatened = sum([1 for x in positions for y in opponents if (x[1] == y[1]+1 or x[1] == y[1]-1) and ((x[0]==y[0]+1 and otherPlayer==1)or (x[0]+1==y[0] and otherPlayer==2))])
        stolenPieces = 2*board.numCols - board.getRemainingPieces()[otherPlayer - 1]
        value = 50 * stolenPieces - 10*numThreatened + max(map(lambda x: board.numRows - 1 - x, board.getPieceDistancesFromGoal()))**2
        #+ sum(map(lambda x: board.numRows - 1 - x, board.getPieceDistancesFromGoal()))
        return (value, None, None)
    # Practically infinity for win and -infinity for loss
    return (1000000 if value == currentPlayer else -1000000, None, None)

def defensiveEvalFn(board, currentPlayer):
    value = board.isGameOver()
    if value == 0:
        # Use weighted heuristics
        # 1 points for every piece not captured
        # 2 point for every opponent piece blocked from forward movement
        # 0 to -(numRows-1) points for each opponent piece closer to the goal
        otherPlayer = 1 if currentPlayer == 2 else 2
        remainingPieces = board.getRemainingPieces()[currentPlayer - 1]
        positions = board.getPiecePositions()
        opponents = board.getOpponentPiecePositions()
        #distances = [dist(pos, opp) for pos in positions for opp in opponents]
        numThreatened = sum([1 for x in positions for y in opponents if (x[1] == y[1]+1 or x[1] == y[1]-1) and ((x[0]==y[0]+1 and otherPlayer==1)or (x[0]+1==y[0] and otherPlayer==2))])
        opponentsToGoal = board.getOpponentPieceDistancesFromGoal()
        closest = board.numRows - 1 - min(opponentsToGoal)
        stolenPieces = 2*board.numCols - board.getRemainingPieces()[otherPlayer - 1]
        value = 10*stolenPieces + 50*remainingPieces - closest**2 - numThreatened*5
        #remainingPieces + 2 * sum(distances) - sum(map(lambda x: 7 - x, opponentsToGoal))
        return (value, None, None)
    # Practically infinity for win and -infinity for loss
    return (1000000 if value == currentPlayer else -1000000, None, None)

def getChildren(board):
    children = []
    for pos in board.getPiecePositions():
        children.append((board.moveForward, 'forward', pos))
        children.append((board.moveDiagonalLeft, 'left', pos))
        children.append((board.moveDiagonalRight, 'right', pos))
    shuffle(children)
    return children

def selectMax(a,b):
   if a == b:
      if(random()>=0.5):
         return a
      return b
   return max(a,b)
   
def selectMin(a,b):
   if a == b:
      if(random()>=0.5):
         return a
      return b
   return min(a,b)

# Note: Found pseudocode with additional information located at:
# https://www.cs.cornell.edu/courses/cs312/2002sp/lectures/rec21.htm
def minimax(board, currentPlayer, evalFn, depth):
    global EXPANDED_COUNT
    leaf = board.isGameOver()
    if leaf != 0 or depth == MAX_DEPTH[currentPlayer - 1]:
        return evalFn(board, currentPlayer)
    EXPANDED_COUNT[currentPlayer - 1] += 1
    children = getChildren(board)
    # Max node:
    if board.playerTurn == currentPlayer:
        v = (-1000000, None, None)
        for child in children:
            (move, dir, pos) = child
            if move(*pos):
                score = minimax(board, currentPlayer, evalFn, depth + 1)[0]
                vp = (score, dir, pos)
                v = selectMax(v, vp)
                board.undoLastMove()
        return v
    else: # Min node
        v = (1000000, None, None)
        for child in children:
            (move, dir, pos) = child
            if move(*pos):
                score = minimax(board, currentPlayer, evalFn, depth + 1)[0]
                vp = (score, dir, pos)
                v = selectMin(v, vp)
                board.undoLastMove()
        return v

def alphabetaPruning(board, currentPlayer, alpha, beta, evalFn, depth):
    global EXPANDED_COUNT
    leaf = board.isGameOver()
    if leaf != 0 or depth == MAX_DEPTH[currentPlayer - 1]:
        return evalFn(board, currentPlayer)
    EXPANDED_COUNT[currentPlayer - 1] += 1
    children = getChildren(board)
    # Max node:
    if board.playerTurn == currentPlayer:
        v = (-1000000, None, None)
        for child in children:
            (move, dir, pos) = child
            if move(*pos):
                score = alphabetaPruning(board, currentPlayer, alpha, beta, evalFn, depth + 1)[0]
                vp = (score, dir, pos)
                v = selectMax(v, vp)
                alpha = max(alpha, v[0])
                board.undoLastMove()
                if beta < alpha:
                    break
        return v
    else: # Min node
        v = (1000000, None, None)
        for child in children:
            (move, dir, pos) = child
            if move(*pos):
                score = alphabetaPruning(board, currentPlayer, alpha, beta, evalFn, depth + 1)[0]
                vp = (score, dir, pos)
                v = selectMin(v, vp)
                beta = min(beta, v[0])
                board.undoLastMove()
                if beta < alpha:
                    break
        return v

def minimaxStrat(board, currentPlayer, evalFn):
    return minimax(board, currentPlayer, evalFn, 0)

def alphabetaStrat(board, currentPlayer, evalFn):
    return alphabetaPruning(board, currentPlayer, -1000000, 1000000, evalFn, 0)

def main():
    global MAX_DEPTH
    args = parseArgs()
    player1Strat = minimaxStrat if args.player1_strat == 'minimax' else alphabetaStrat
    player2Strat = minimaxStrat if args.player2_strat == 'minimax' else alphabetaStrat
    eval1Fn = offensiveEvalFn if args.player1_eval == 'offensive' else defensiveEvalFn
    eval2Fn = offensiveEvalFn if args.player2_eval == 'offensive' else defensiveEvalFn
    MAX_DEPTH[0] = args.player1_max_depth
    MAX_DEPTH[1] = args.player2_max_depth
    numWTB = args.num_workers_to_base if args.num_workers_to_base > 1 else 1
    boardType = args.rectangle_board  # if true, rectangle 5x10, else square 8x8
    numPcsPerPlayer = 20 if (boardType) else 16

    # Initialize board
    board = BreakthroughBoard(numWTB, boardType)

    # Define strategies for player 1 and player 2
    strats = [(player1Strat, eval1Fn), (player2Strat, eval2Fn)]
    last = 2
    turns = 0
    playerTurns = [0, 0]
    playerThinkTime = [0, 0]
    currentTimeMillis = lambda: int(round(time.time() * 1000))
    print "Playing..."
    while not board.isGameOver():
        player = board.playerTurn
        # Paranoia, ensure alternating turns
        assert player != last
        last = player
        startThink = currentTimeMillis()
        (strat, evalFn) = strats[player - 1]
        result = strat(board, player, evalFn)
        endThink = currentTimeMillis()
        totalThinkTime = endThink - startThink
        playerThinkTime[player - 1] += totalThinkTime
        (score, direction, pos) = result
        if direction == 'forward':
            board.moveForward(*pos)
        elif direction == 'left':
            board.moveDiagonalLeft(*pos)
        elif direction == 'right':
            board.moveDiagonalRight(*pos)
        else:
            print "Something went wrong: {}".format(result)
            assert False
        playerTurns[player - 1] += 1
        # Show progress every 5 turns
        # if turns % 5 == 0:
        #    print "-----"
        #    board.printBoard()
        turns += 1
    print ""
    print "Game complete!"
    print "-----"
    print "Final board configuration:"
    board.printBoard()
    print ""
    print "Stats:"
    print "------"
    print "Expanded nodes for player 1: {}".format(EXPANDED_COUNT[0])
    print "Expanded nodes for player 2: {}".format(EXPANDED_COUNT[1])
    print "Average expanded nodes per move for player 1: {}".format(float(EXPANDED_COUNT[0]) / float(playerTurns[0]))
    print "Average expanded nodes per move for player 2: {}".format(float(EXPANDED_COUNT[1]) / float(playerTurns[1]))
    print "Average time per move for player 1: {}ms".format(float(playerThinkTime[0]) / float(playerTurns[0]))
    print "Average time per move for player 2: {}ms".format(float(playerThinkTime[1]) / float(playerTurns[1]))
    print "Pieces captured by player 1: {}".format(numPcsPerPlayer - board.getRemainingPieces()[1])
    print "Pieces captured by player 2: {}".format(numPcsPerPlayer - board.getRemainingPieces()[0])
    print "Total turns taken throughout game: {}".format(turns)
    print "Game winner: {}".format(board.isGameOver())

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s1', '--player1-strat', choices=('alphabeta', 'minimax'), required=True)
    parser.add_argument('-s2', '--player2-strat', choices=('alphabeta', 'minimax'), required=True)
    parser.add_argument('-e1', '--player1-eval', choices=('offensive', 'defensive'), required=True)
    parser.add_argument('-e2', '--player2-eval', choices=('offensive', 'defensive'), required=True)
    parser.add_argument('-d1', '--player1-max-depth', type=int, required=True)
    parser.add_argument('-d2', '--player2-max-depth', type=int, required=True)
    parser.add_argument('-wtb', '--num-workers-to-base', type=int, required=False)  # default is 1
    parser.add_argument('-rect', '--rectangle-board', action="store_true", default=False)

    return parser.parse_args()

def logicalTest():
    """ Logical test to run through and observe as human to verify board """
    board = BreakthroughBoard()
    print board.moveForward(6, 1)
    board.endTurn()
    print board.moveForward(1, 1)
    board.endTurn()
    print board.moveDiagonalLeft(2, 2)
    board.endTurn()
    print board.moveDiagonalLeft(1, 0)
    print board.moveForward(6, 1)
    board.endTurn()
    print board.moveForward(5, 1)
    board.endTurn()
    print board.moveForward(4, 1)
    board.endTurn()
    print board.moveForward(3, 1)
    board.endTurn()
    print board.getRemainingPieces()
    print board.moveForward(1, 0)
    board.endTurn()
    print board.moveDiagonalLeft(2, 0)
    board.endTurn()
    print board.getRemainingPieces()
    print board.moveDiagonalRight(7, 0)
    print board.moveDiagonalLeft(6, 0)
    board.printBoard()
    print "--------"
    # Undo everything...
    for i in range(len(board.moveHistory)):
        board.undoLastMove()
    board.printBoard()
    print board.remainingPieces

if __name__ == "__main__":
    main()
