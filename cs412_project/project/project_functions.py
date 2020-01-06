# imports
import math
import numpy
import random

# these checks will be easy; we know the nature of each variable in our data.
def IsContinuous(index):
   return index in [3,7,8,9,10,11,14,16,28,33,34,35,36]
def IsOrdinal(index):
   return index in [37,46,51,60,68]

# Returns the value at index 'index' that occurs the most frequently 
#  at 'index' in each entry in 'Data'
def GetMaxCount(Data, index):
   counts = {}
   for e in Data:
      if e[index] not in counts:
         counts[e[index]] = 0
      counts[e[index]] += 1
   maxvalue = ''
   maxcount = 0
   for e in counts:
      if counts[e] > maxcount:
         maxcount = counts[e]
         maxindex = e
   return maxindex

# Returns the ordinal average value for each entry in 'Data' at 'index'
def GetOrdinalAverage(Data, index):
   count = 0
   sum = 0.0
   for e in Data:
      if e[index] != '':
         count += 1
         sum += float(e[index])
   return round(sum/float(count))

# returns the continuous variable average for each entry in 'Data' at 'index'
def GetAverageValue(Data, index):
   count = 0
   sum = 0.0
   for e in Data:
      if e[index] != '':
         count += 1
         sum += float(e[index])
   return sum/float(count)

# gets the average value of 'Data' at 'index' given that entry of Data has 'result' == 'e'
def GetAverageValueAt(Data,Results,index,e):
   sum = 0.0
   count = 0
   for i in range(len(Data)):
      if Results[i] != e:
         continue
      sum += float(Data[i][index])
      count += 1
   return sum/float(count)

# gets the std dev value of 'Data' at 'index' given that entry of Data has 'result' == 'e'
def GetStdDevAt(Data,Results,index,e):
   values = []
   for i in range(len(Data)):
      if Results[i] != e:
         continue
      values.append(float(Data[i][index]))
   return numpy.std(values)

# gets the counts of each nominal value of 'Data' at 'index' given that entry of Data has 'result' == 'e'
def GetCountsAt(Data,Results,index,e):
   counts = {}
   for i in range(len(Data)):
      if Results[i] != e:
         continue
      if Data[i][index] not in counts:
         counts[Data[i][index]] = 0
      counts[Data[i][index]] += 1
   return counts

# This function creates an object that is intended for use in filling in gaps in data sets.
# DataSets, the return object, should consist of one entry for each field in question
def GenerateDefaultValues(Data):
   DataSets = []
   for index in range(len(Data[0])):
      DataSets.append('')
      if IsContinuous(index): 
         DataSets[index] = GetAverageValue(Data,index) 
      elif IsOrdinal(index): 
         DataSets[index] = GetOrdinalAverage(Data,index)
      else: # is a nominal attribute (most are this)
         DataSets[index] = GetMaxCount(Data,index) 
   return DataSets

# This function will fill gaps in provided data.
# The return value, Data, contains many entries, all of which have data in every possible field.
def FillMissingData(Data):
   print("Filling Missing Data...")
   DataSets = GenerateDefaultValues(Data)
   for entry in Data:
      for index in range(len(entry)):
         if entry[index] == '':
            entry[index] = DataSets[index]
   return Data

class NBEntry:
   def __init__(self):
      self.attributeType = ''
      self.mean = 0
      self.stddev = 0
      self.counts = {}
   def __str__(self):
      if self.attributeType == 'Nominal':
         return self.attributeType+'  '+str(self.counts)
      return self.attributeType+'  '+str(self.mean)+':'+str(self.stddev)

# this function creates index-based data necessary for Naive Bayes model
def NaiveBayesianClassifier(Data, Results):
   ResultValues = []
   for e in Results:
      if e not in ResultValues:
         ResultValues.append(e)
   ResultValues.sort()

   # this classifier is a data structure of the form 
   #  {result:[(entry),(entry)...],result2:[(entry),...]}
   # where the number of entries in the list corresponding to the result value is equal
   # to the number of variables in the dataset
   # entries are different based on variable type: continuous and ordinal values have 
   # mean and stdDev, where nominal values have counts of each nominal value in a dictionary
   SummarizedData = {}
   for e in ResultValues:
      SummarizedData[e] = []
      for index in range(len(Data[0])):
         SummarizedData[e].append(NBEntry())
         if IsContinuous(index): 
            SummarizedData[e][-1].attributeType = 'Continuous'
            SummarizedData[e][-1].mean = GetAverageValueAt(Data,Results,index,e) 
            SummarizedData[e][-1].stddev = GetStdDevAt(Data,Results,index,e) 
         elif IsOrdinal(index): 
            SummarizedData[e][-1].attributeType = 'Ordinal'
            SummarizedData[e][-1].mean = round(GetAverageValueAt(Data,Results,index,e)) 
            SummarizedData[e][-1].stddev = GetStdDevAt(Data,Results,index,e)
         else: # is a nominal attribute (most are this)
            SummarizedData[e][-1].attributeType = 'Nominal'
            SummarizedData[e][-1].counts = GetCountsAt(Data,Results,index,e)
   return SummarizedData

def CreateClassifier(Data, Results):
   print("Creating Classifier...")
   Classifier = NaiveBayesianClassifier(Data,Results)
   return Classifier

def PickRandomValues(start,end,count):
   retVal = []
   for e in range(count):
      retVal.append(random.randint(start,end))
   return retVal

def TrimTrainingData(TrainingData,Results):
   print("Trimming training data so that each class has equal representation...")
   counts = {}
   for e in range(len(TrainingData)):
      if Results[e] not in counts:
         counts[Results[e]] = 0
      counts[Results[e]] += 1
   minVal = len(TrainingData)
   for e in counts:
      if counts[e] < minVal:
         minVal = counts[e]
   for e in counts:
      counts[e] = minVal
   newData = []
   newResults = []
   for e in range(len(TrainingData)):
      if counts[Results[e]] > 0:
         newData.append(TrainingData[e])
         newResults.append(Results[e])
         counts[Results[e]]-=1
   return newData,newResults

def GetMode(Values):
   maxCount = 0
   maxResult = 0
   for x in Values:
      if Values[x] > maxCount:
         maxCount = Values[x]
         maxResult = x
   return maxResult

# wrapper for distance and result so that they can be ordered
class DistanceVal:
   def __init__(self,distance,val):
      self.distance = distance
      self.val = val

def GetDistance(One,Two,Classification,Weights):
   distance = 0.0
   for e in range(len(Classification)):
      if Classification[e].attributeType == 'Nominal': 
         if One[e] != Two[e]:
            distance += 1.0*Weights[e]
      else: # continuous or nominal - can use std dev for this comparison
         if isfloat(One[e]) and isfloat(Two[e]):
            distance += (float(One[e])-float(Two[e]))/Classification[e].stddev * Weights[e]
   return distance

def GetKClosestValues(K,N,Training,Test,Results,Classifier,Weights):
   Index = len(Training)
   kMost = []
   x = 0
   # initially fill stack so that replacement doesn't need to know about empty entries
   for k in range(K):
      kMost.append(DistanceVal(999999999999,0))
   randomValues = PickRandomValues(0,len(Training)-1,N)
   if N == len(Training):
      randomValues = range(len(Training))
   for n in randomValues:
      dist = GetDistance(Training[n],Test,Classifier[Results[n]],Weights)
      if dist < kMost[-1].distance:
         x = K-1
         while dist < kMost[x].distance and x>0:
            x -= 1
         kMost.pop()
         kMost.insert(x,DistanceVal(dist,Results[n]))
   retVal = {}
   for e in kMost:
      if e.val not in retVal:
         retVal[e.val] = 0
      retVal[e.val]+=1
   return retVal

# from HW5
def OverallEntropy(Results,DataLen):
   results = {}
   for e in range(DataLen):
      if Results[e] not in results:
         results[Results[e]] = 0
      results[Results[e]] += 1
   entropy = 0.0
   for e in results:
      p_i = float(results[e])/float(DataLen)
      entropy += -1.0 * p_i * math.log2(p_i)
   return entropy

# from HW5
def InformationGain(index, data, Results, entropy):
   sumEntropies = 0.0
   results = {}
   for e in range(len(data)):
      if data[e][index] not in results:
         results[data[e][index]] = {}
      if Results[e] not in results[data[e][index]]:
         results[data[e][index]][Results[e]] = 0
      results[data[e][index]][Results[e]] += 1
   # results contains a dict for each attribute value that contains 
   # counts of each result corresponding to that attribute value
   for e in results:
      iEntropy = 0.0
      sum = 0
      for f in results[e]:
         sum += results[e][f]
      for f in results[e]:
         p_i = float(results[e][f])/float(sum)
         iEntropy += -1.0 * p_i * math.log2(p_i)
      sumEntropies += iEntropy * float(sum)/float(len(data))
   return entropy - sumEntropies

def DetermineWeightOfAttribute(TrainingData,AttributeIndex,Results,Entropy):
   return InformationGain(AttributeIndex,TrainingData,Results,Entropy)

def GenerateWeights(TrainingData,Results):
   retVal = []
   print("Generating weights of attributes...")
   entropy = OverallEntropy(Results,len(TrainingData))
   for e in range(len(TrainingData[0])):
      retVal.append(DetermineWeightOfAttribute(TrainingData,e,Results,entropy))
   return retVal

def KNNClassifier(Classifier, Training, Testing, Index, Results):
   Weights = GenerateWeights(Training,Results)
   print("Running KNN with k=30 and 1000-sample training sets with replacement...")
   for x in range(len(Testing)):
      Values = GetKClosestValues(30,1000,Training,Testing[x],Results,Classifier,Weights)
      result = GetMode(Values)
      print("Classification for test entry",x,"is",result,"from",Values)
      Results[Index+x] = result
   return Results

def isfloat(value):
  try:
    float(value)
    return True
  except:
    return False