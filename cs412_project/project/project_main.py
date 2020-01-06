# imports
import sys
import math
import operator
import project_functions as pf

# Interpret the arguments passed into the command line
traindata = "../results/training.csv"
if len(sys.argv) > 1:
   traindata = sys.argv[1]
testdata = "../results/testing.csv"
if len(sys.argv) > 2:
   testdata = sys.argv[2]
outfile = "../results/submission.csv"
if len(sys.argv) > 3:
   outfile = sys.argv[3]

#globals
TrainingData = [] # contains input data
TestingData = [] # contains second set of data (for testing)
Labels = [] # contains names of fields in data
Fields = 0 # number of fields (excluding results from testing data)
CategoryTypes = [] # contains the types of variables, used in normalization

testingIndex = 0 # id at which testing data starts - used for output
Results = [] # list of results from full data
ResultsName = "" # name of results field
IDs = [] # list of id numbers from full data
IDName = "" # name of id field

# read in values from training.csv
def ingestTrainingData(source):
   global TrainingData, Labels, Results, ResultsName, IDs, IDName
   linenumber = 0
   with open(source) as f:
      for line in f:
         if linenumber == 0:
             Labels = line.strip().split(',')
             IDName = Labels[0]
             Labels.remove(Labels[0])
             ResultsName = Labels[-1]
             Labels.remove(Labels[-1])
         if linenumber >= 1:
            TrainingData.append(line.strip().split(',')) # missing data will be empty strings
            IDs.append(TrainingData[-1][0])
            TrainingData[-1].remove(TrainingData[-1][0]) # ID should not be a classifiable field
            Results.append(TrainingData[-1][-1])
            TrainingData[-1].remove(TrainingData[-1][-1]) # result should not be a classifiable field
         linenumber = linenumber+1

# read in values from testing.csv
def ingestTestingData(source):
   global TestingData, testingIndex
   linenumber = 0
   with open(source) as f:
      for line in f:
         if linenumber == 1:
            testingIndex = int(line.strip().split(',')[0])
         if linenumber >= 1:
            Results.append('1')
            TestingData.append(line.strip().split(','))
            IDs.append(TestingData[-1][0])
            TestingData[-1].remove(TestingData[-1][0]) # ID should not be a classifiable field
         linenumber = linenumber+1

def makeOutput(output):
   global Results, IDs, originalTestingIndex, testingIndex
   with open(output,"w+") as fi:
      fi.write(IDName+','+ResultsName+'\n')
      for f in range(len(Results)-testingIndex):
         fi.write(IDs[f+originalTestingIndex]+','+Results[f+testingIndex]+'\n')
   print("Done!")

ingestTrainingData(traindata)
ingestTestingData(testdata)

# do some stuff
TrainingData = pf.FillMissingData(TrainingData)
originalTestingIndex = len(TrainingData)
Classifier = pf.CreateClassifier(TrainingData, Results) # still want proportional data in the classifier
TrainingData,Results = pf.TrimTrainingData(TrainingData, Results)
for e in range(len(TestingData)):
   Results.append('1')
testingIndex = len(TrainingData)
Results = pf.KNNClassifier(Classifier,TrainingData,TestingData,testingIndex,Results)

makeOutput(outfile)

