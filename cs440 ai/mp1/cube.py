'''  Solution 1:
r r b b g g 
r r b b g g 
    o o 
    o o 
    y y 
    y y 
    p p 
    p p
'''
import math
class Cube:
   def __init__(self, solnCubes=None, generate=False): # all sides stored separately from solution perspective; topleft, clockwise
      self.left = ['','','','']
      self.top = ['','','','']
      self.right = ['','','','']
      self.front = ['','','','']
      self.bottom = ['','','','']
      self.back = ['','','','']
      self.path = []
      self.score = 0
      self._numberWrong = -1
      self.solutionCubes = self.generateSolutions() if generate else solnCubes
   
   # Check if two cubes have the same state
   def __eq__(self,other):
      retVal = True
      for each in range(0,4):
         retVal &= self.left[each] == other.left[each]
         retVal &= self.right[each] == other.right[each]
         retVal &= self.top[each] == other.top[each]
         retVal &= self.bottom[each] == other.bottom[each]
         retVal &= self.front[each] == other.front[each]
         retVal &= self.back[each] == other.back[each]
      return retVal
   
   # Compare to path scores
   def __cmp__(self,other):
      return cmp(self.score, other.score)
   
   # Return a hash of the path
   def __hash__(self):
        return hash(str(self.path))
   
   # Run the heuristic with divisible factor of 8
   def Heuristic(self):
      return self.numberWrong() / 8
  
   # Update the score for the current solution path
   def updateScore(self):
    self.score = len(self.path) + self.Heuristic()


   # Open file and parse the initial face layouts
   def load(self,filename):
      """
      Labels sides in clockwise fashion as follows:
        r g
        b o
      Would result in:
        side[0] = r
        side[1] = g
        side[2] = o
        side[3] = b
      """
      with open(filename) as f:
         lineNum=0
         for line in f:
            a = line.split(' ')
            if lineNum == 0:
               self.left[0] = a[0]
               self.left[1] = a[1]
               self.top[0] = a[2]
               self.top[1] = a[3]
               self.right[0] = a[4]
               self.right[1] = a[5].strip()
            elif lineNum == 1:
               self.left[3] = a[0]
               self.left[2] = a[1]
               self.top[3] = a[2]
               self.top[2] = a[3]
               self.right[3] = a[4]
               self.right[2] = a[5].strip()
            elif lineNum == 2:
               self.front[0] = a[4]
               self.front[1] = a[5].strip()
            elif lineNum == 3:
               self.front[3] = a[4]
               self.front[2] = a[5].strip()
            elif lineNum == 4:
               self.bottom[0] = a[4]
               self.bottom[1] = a[5].strip()
            elif lineNum == 5:
               self.bottom[3] = a[4]
               self.bottom[2] = a[5].strip()
            elif lineNum == 6:
               self.back[0] = a[4]
               self.back[1] = a[5].strip()
            elif lineNum == 7:
               self.back[3] = a[4]
               self.back[2] = a[5].strip()
            lineNum=1+lineNum
      self.score = self.Heuristic()

   # Display formatted output of the current cube state to the console
   def show(self):
      print self.left[0],self.left[1],self.top[0],self.top[1],self.right[0],self.right[1]
      print self.left[3],self.left[2],self.top[3],self.top[2],self.right[3],self.right[2]
      print '   ',                  self.front[0],self.front[1]
      print '   ',                  self.front[3],self.front[2]
      print '   ',                 self.bottom[0],self.bottom[1]
      print '   ',                 self.bottom[3],self.bottom[2]
      print '   ',                   self.back[0],self.back[1]
      print '   ',                   self.back[3],self.back[2]
   def numberWrong(self):
      retVal = self._numberWrong
      if retVal == -1:
         for soln in self.solutionCubes:
            wrongCount = 0
            for each in range(4):
              if self.left[each]!=soln.left[each]:
                 wrongCount+=1
              if self.right[each]!=soln.right[each]:
                 wrongCount+=1
              if self.top[each]!=soln.top[each]:
                wrongCount+=1
              if self.bottom[each]!=soln.bottom[each]:
                 wrongCount+=1
              if self.front[each]!=soln.front[each]:
                 wrongCount+=1
              if self.back[each]!=soln.back[each]:
                 wrongCount+=1
            retVal = min(wrongCount, retVal) if retVal != -1 else wrongCount
         self._numberWrong = retVal
      return retVal
   
   # Checks if solution is one or two turns away 
   def isSolutionOne(self):
      return (self.numberWrong() == 0)
   def isSolutionTwo(self):
      retVal = True
      for each in range(0,3):
         retVal &= self.left[each]==self.left[3]
         retVal &= self.right[each]==self.right[3]
         retVal &= self.top[each]==self.top[3]
         retVal &= self.bottom[each]==self.bottom[3]
         retVal &= self.front[each]==self.front[3]
         retVal &= self.back[each]==self.back[3]
      return retVal
   
   # Creates a new copy of the current cube
   def copyTo(self, retVal):
      retVal.path = []
      for each in range(0,4):
         retVal.left[each] = self.left[each]
         retVal.right[each] = self.right[each]
         retVal.top[each] = self.top[each]
         retVal.bottom[each] = self.bottom[each]
         retVal.front[each] = self.front[each]
         retVal.back[each] = self.back[each]
      for each in self.path:
         retVal.path.append(each)
      retVal.score = self.score

   #
   # Following functions represent the 12 different possible
   # rotations (6 faces with each having clockwise and 
   # counter-clockwise rotations)
   #
   def rotateLeftClockwise(self):
      tmp = self.left[0]
      self.left[0] = self.left[3]
      self.left[3] = self.left[2]
      self.left[2] = self.left[1]
      self.left[1] = tmp
      tmp = self.top[0]
      tmp2 = self.top[3]
      self.top[0] = self.back[0]
      self.top[3] = self.back[3]
      self.back[0] = self.bottom[0]
      self.back[3] = self.bottom[3]
      self.bottom[0] = self.front[0]
      self.bottom[3] = self.front[3]
      self.front[0] = tmp
      self.front[3] = tmp2
      self.path.append("L")
   def rotateLeftCounterclockwise(self):
      self.rotateLeftClockwise()
      self.rotateLeftClockwise()
      self.rotateLeftClockwise()
      self.path.pop()
      self.path.pop()
      self.path.pop()
      self.path.append("L'")
   def rotateRightClockwise(self):
      tmp = self.right[0]
      self.right[0] = self.right[3]
      self.right[3] = self.right[2]
      self.right[2] = self.right[1]
      self.right[1] = tmp         
      tmp = self.top[1]
      tmp2 = self.top[2]
      self.top[1] = self.front[1]
      self.top[2] = self.front[2]
      self.front[1] = self.bottom[1]
      self.front[2] = self.bottom[2]
      self.bottom[1] = self.back[1]
      self.bottom[2] = self.back[2]
      self.back[1] = tmp
      self.back[2] = tmp2
      self.path.append("R")
   def rotateRightCounterclockwise(self):
      self.rotateRightClockwise()
      self.rotateRightClockwise()
      self.rotateRightClockwise()
      self.path.pop()
      self.path.pop()
      self.path.pop()
      self.path.append("R'")
   def rotateTopClockwise(self):
      tmp = self.top[0]
      self.top[0] = self.top[3]
      self.top[3] = self.top[2]
      self.top[2] = self.top[1]
      self.top[1] = tmp
      tmp = self.right[0]
      tmp2 = self.right[3]
      self.right[0] = self.back[3]
      self.right[3] = self.back[2]
      self.back[2] = self.left[1]
      self.back[3] = self.left[2]
      self.left[1] = self.front[0]
      self.left[2] = self.front[1]
      self.front[0] = tmp2
      self.front[1] = tmp
      self.path.append("T")
   def rotateTopCounterclockwise(self):
      self.rotateTopClockwise()
      self.rotateTopClockwise()
      self.rotateTopClockwise()
      self.path.pop()
      self.path.pop()
      self.path.pop()
      self.path.append("T'")
   def rotateBottomClockwise(self):
      tmp = self.bottom[0]
      self.bottom[0] = self.bottom[3]
      self.bottom[3] = self.bottom[2]
      self.bottom[2] = self.bottom[1]
      self.bottom[1] = tmp
      tmp = self.back[0]
      tmp2 = self.back[1]
      self.back[0] = self.right[1]
      self.back[1] = self.right[2]
      self.right[1] = self.front[2]
      self.right[2] = self.front[3]
      self.front[2] = self.left[3]
      self.front[3] = self.left[0]
      self.left[3] = tmp
      self.left[0] = tmp2
      self.path.append("Bo")
   def rotateBottomCounterclockwise(self):
      self.rotateBottomClockwise()
      self.rotateBottomClockwise()
      self.rotateBottomClockwise()
      self.path.pop()
      self.path.pop()
      self.path.pop()
      self.path.append("Bo'")
   def rotateFrontClockwise(self):
      tmp = self.front[0]
      self.front[0] = self.front[3]
      self.front[3] = self.front[2]
      self.front[2] = self.front[1]
      self.front[1] = tmp
      tmp = self.top[2]
      tmp2 = self.top[3]
      self.top[2] = self.left[2]
      self.top[3] = self.left[3]
      self.left[2] = self.bottom[0]
      self.left[3] = self.bottom[1]
      self.bottom[0] = self.right[2]
      self.bottom[1] = self.right[3]
      self.right[2] = tmp
      self.right[3] = tmp2
      self.path.append("F")
   def rotateFrontCounterclockwise(self):
      self.rotateFrontClockwise()
      self.rotateFrontClockwise()
      self.rotateFrontClockwise()
      self.path.pop()
      self.path.pop()
      self.path.pop()
      self.path.append("F'")
   def rotateBackClockwise(self):
      tmp = self.back[3]
      self.back[3] = self.back[0]
      self.back[0] = self.back[1]
      self.back[1] = self.back[2]
      self.back[2] = tmp
      tmp = self.top[0]
      tmp2 = self.top[1]
      self.top[0] = self.right[0]
      self.top[1] = self.right[1]
      self.right[0] = self.bottom[2]
      self.right[1] = self.bottom[3]
      self.bottom[2] = self.left[0]
      self.bottom[3] = self.left[1]
      self.left[0] = tmp
      self.left[1] = tmp2
      self.path.append("Ba")
   def rotateBackCounterclockwise(self):
      self.rotateBackClockwise()
      self.rotateBackClockwise()
      self.rotateBackClockwise()
      self.path.pop()
      self.path.pop()
      self.path.pop()
      self.path.append("Ba'")
   def Successors(self):
      retVal = []
      tmp = Cube(self.solutionCubes)
      self.copyTo(tmp)
      tmp.rotateLeftClockwise()
      retVal.append(tmp)
      tmp = Cube(self.solutionCubes)
      self.copyTo(tmp)
      tmp.rotateLeftCounterclockwise()
      retVal.append(tmp)
      tmp = Cube(self.solutionCubes)
      self.copyTo(tmp)
      tmp.rotateRightClockwise()
      retVal.append(tmp)
      tmp = Cube(self.solutionCubes)
      self.copyTo(tmp)
      tmp.rotateRightCounterclockwise()
      retVal.append(tmp)
      tmp = Cube(self.solutionCubes)
      self.copyTo(tmp)
      tmp.rotateTopClockwise()
      retVal.append(tmp)
      tmp = Cube(self.solutionCubes)
      self.copyTo(tmp)
      tmp.rotateTopCounterclockwise()
      retVal.append(tmp)
      tmp = Cube(self.solutionCubes)
      self.copyTo(tmp)
      tmp.rotateBottomClockwise()
      retVal.append(tmp)
      tmp = Cube(self.solutionCubes)
      self.copyTo(tmp)
      tmp.rotateBottomCounterclockwise()
      retVal.append(tmp)
      tmp = Cube(self.solutionCubes)
      self.copyTo(tmp)
      tmp.rotateBackClockwise()
      retVal.append(tmp)
      tmp = Cube(self.solutionCubes)
      self.copyTo(tmp)
      tmp.rotateBackCounterclockwise()
      retVal.append(tmp)
      tmp = Cube(self.solutionCubes)
      self.copyTo(tmp)
      tmp.rotateFrontClockwise()
      retVal.append(tmp)
      tmp = Cube(self.solutionCubes)
      self.copyTo(tmp)
      tmp.rotateFrontCounterclockwise()
      retVal.append(tmp)
      return retVal
   def generateSolutions(self):
       solns = set()
       primary = Cube(solns, False)
       primary.left = ['r' for x in range(4)]
       primary.top = ['b' for x in range(4)]
       primary.right = ['g' for x in range(4)]
       primary.front = ['o' for x in range(4)]
       primary.bottom = ['y' for x in range(4)]
       primary.back = ['p' for x in range(4)]
       solns.add(primary)
       lastRotZ = primary
       for x in range(4): # Rotate cube about z-axis
           rotZ = Cube(solns)
           lastRotZ.copyTo(rotZ)
           rotZ.rotateGlobalAboutZ()
           lastRotT = rotZ
           for y in range(4): # Rotate about the top face
               rotT = Cube(solns)
               lastRotT.copyTo(rotT)
               rotT.rotateGlobalAboutTop()
               lastRotX = rotT
               for z in range(4): # Rotate about x-axis
                   rotX = Cube(solns)
                   lastRotX.copyTo(rotX)
                   rotX.rotateGlobalAboutX()
                   solns.add(rotX)
                   lastRotX = rotX
               lastRotT = rotT
           lastRotZ = rotZ
       return solns
   def rotateGlobalAboutTop(self):
      tmp = self.left
      self.left = self.front
      self.front = self.right
      self.right = self.back
      self.back = tmp
   def rotateGlobalAboutZ(self):
       tmp = self.left
       self.left = self.bottom
       self.bottom = self.right
       self.right = self.top
       self.top = tmp
   def rotateGlobalAboutX(self):
       tmp = self.back
       self.back = self.bottom
       self.bottom = self.front
       self.front = self.top
       self.top = tmp
