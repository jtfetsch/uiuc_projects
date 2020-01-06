"""
BreakthroughBoard.py
Breakthrough game logic
"""

class BreakthroughBoard:
    def __init__(self, numWTB, boardType, copy=None):
        self.board = []
        self.playerTurn = 1
        self.remainingPieces = []
        self.numWorkersToBase = numWTB
        self.boardType = boardType 
        self.numRows = 0
        self.numCols = 0

        # Move history format: (startPos, startVal, movedToPos, movedToVal, playerPieceTaken)
        self.moveHistory = []

        # Player 1 pieces represented by a 1, player 2 pieces represented by a 2,
        # and unoccupied spaces represented by 0
        
        # rectangle 5x10 board size, 2x10 pieces per player 
        if self.boardType:
            self.remainingPieces = [20, 20]
            self.numRows = 5
            self.numCols = 10
        
        else: # square 8x8 board size, 2x8 pieces per player
            self.remainingPieces = [16, 16]
            self.numRows = 8
            self.numCols = 8
        
        for row in range(self.numRows):
            self.board.append([1 if row in (0, 1) else 2 if row in (self.numRows-2, self.numRows-1) else 0 for col in range(self.numCols)])
    
    def moveForward(self, row, col):
        """ Moves a piece at position (row, col) forward one space.
            Returns true if moved successfully, false otherwise.
        """
        # Cannot move pieces that aren't ours
        if not self._isInBounds(row, col) or self.board[row][col] != self.playerTurn:
            return False
        forward = row + 1 if self.playerTurn == 1 else row - 1
        # Make sure we're moving into a position that is (a) on the board and (b) a free space
        if not self._isInBounds(forward, col) or self.board[forward][col] != 0:
            return False
        self.moveHistory.append(((row, col), self.playerTurn, (forward, col), self.board[forward][col], 0))
        self.board[forward][col] = self.board[row][col]
        self.board[row][col] = 0
        self.endTurn()
        return True
    
    def moveDiagonalLeft(self, row, col):
        return self._moveForwardDiagonal(row, col, True)
    
    def moveDiagonalRight(self, row, col):
        return self._moveForwardDiagonal(row, col, False)
    
    def _moveForwardDiagonal(self, row, col, left):
        """ Moves a piece from (row, col) 1 space diagonally forward.
            Returns 0 if move could not be made, 1 if move was successful, and
            2 if move captured an opponent piece.

            NOTE: Left is from the perspective of the player.
        """
        # Cannot move pieces that aren't ours
        if not self._isInBounds(row, col) or self.board[row][col] != self.playerTurn:
            return 0
        forward = row + 1 if self.playerTurn == 1 else row - 1
        horiz = (col + 1 if self.playerTurn == 1 else col - 1) if left else (col - 1 if self.playerTurn == 1 else col + 1)
        # Cannot move to a space we already occupy
        if not self._isInBounds(forward, horiz) or self.board[forward][horiz] == self.playerTurn:
            return 0
        spaceVal = self.board[forward][horiz]
        if spaceVal != 0:
            self.remainingPieces[spaceVal - 1] -= 1
        self.moveHistory.append(((row, col), self.playerTurn, (forward, horiz), self.board[forward][horiz], spaceVal))
        self.board[forward][horiz] = self.playerTurn
        self.board[row][col] = 0
        self.endTurn()
        return 1 if spaceVal == 0 else 2
    
    def getPiecePositions(self):
        """ Get the piece positions for the current player """
        return self._getPiecePositions(self.playerTurn)
    
    def getOpponentPiecePositions(self):
        """ Get the piece positions for the opponent player """
        return self._getPiecePositions(self._getOtherPlayer())

    def _getPiecePositions(self, player):
        """ Get the piece positions for some player """
        positions = []
        for row in range(self.numRows):
            for col in range(self.numCols):
                if self.board[row][col] == player:
                    positions.append((row, col))
        return positions
    
    def _getOtherPlayer(self):
        return 2 if self.playerTurn == 1 else 1
    
    def getPieceDistancesFromGoal(self):
        return self._getPieceDistancesFromGoal(False)
    
    def getOpponentPieceDistancesFromGoal(self):
        return self._getPieceDistancesFromGoal(True)

    def _getPieceDistancesFromGoal(self, opponent=False):
        """ Get a list of piece positions from the goal for current player """
        positions = self.getOpponentPiecePositions() if opponent else self.getPiecePositions()
        player = self._getOtherPlayer() if opponent else self.playerTurn
        if player == 1:
            return [(self.numRows-1) - row for (row, col) in positions]
        return [row for (row, col) in positions]
    
    def undoLastMove(self):
        if len(self.moveHistory) == 0:
            return False
        ((origRow, origCol), origVal, (movedRow, movedCol), movedVal, taken) = self.moveHistory.pop()
        # Restore taken players
        if taken != 0:
            self.remainingPieces[taken - 1] += 1
        # Restore previous configuration:
        self.board[origRow][origCol] = origVal
        self.board[movedRow][movedCol] = movedVal
        # Restore turn
        self.playerTurn = 1 if self.playerTurn == 2 else 2
    
    def isGameOver(self):
        """ Returns the identity of the winning player if game is over.
            Otherwise, returns 0.
        """
        if self.remainingPieces[1] == 0 or self.board[self.numRows-1].count(1) == self.numWorkersToBase:
            return 1
        if self.remainingPieces[0] == 0 or self.board[0].count(2) == self.numWorkersToBase:
            return 2
        return 0
    
    def getRemainingPieces(self):
        return self.remainingPieces
    
    def endTurn(self):
        """ Ends a player's move """
        self.playerTurn = 1 if self.playerTurn == 2 else 2
    
    def _isInBounds(self, row, col):
        return row >= 0 and row < self.numRows and col >= 0 and col < self.numCols

    def printBoard(self):
        for row in self.board:
            for col in row:
                print col,
            print ""
