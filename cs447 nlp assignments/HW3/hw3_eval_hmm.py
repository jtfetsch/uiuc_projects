##
## Part 1:
## Evaluate the output of your bigram HMM POS tagger
##
import os.path
import sys
from operator import itemgetter


# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_');
        self.word = parts[0]
        self.tag = parts[1]

# Reads a labeled data inputFile, and returns a nested list of sentences, where each sentence is a list of TaggedWord objects
def readLabeledData(inputFile):
    if os.path.isfile(inputFile):
        file = open(inputFile, "r")  # open the input file in read-only mode
        sens = [];
        for line in file:
            raw = line.split()
            sentence = []
            for token in raw:
                sentence.append(TaggedWord(token))
            sens.append(sentence)  # append this list as an element to the list of sentences
        return sens
    else:
        print("Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
        sys.exit()  # exit the script

# A class for evaluating POS-tagged data
class Eval:
    ################################
    #intput:                       #
    #    goldFile: string          #
    #    testFile: string          #
    #output: None                  #
    ################################
    def __init__(self, goldFile, testFile):
        self.goldTagSequence = readLabeledData(goldFile)
        self.testTagSequence = readLabeledData(testFile)
        self.tagSet = set()
        self.tagList = []
        self.confusionMatrix = {}

        # check for length mismatches, record set of tags
        if len(self.goldTagSequence) != len(self.testTagSequence):
            print("Data length mismatch")
        for sentenceIndex in range(len(self.goldTagSequence)):
            if len(self.goldTagSequence[sentenceIndex]) != len(self.testTagSequence[sentenceIndex]):
                print("Data length mismatch")
            for wordIndex in range(len(self.goldTagSequence[sentenceIndex])):
                goldTag = self.goldTagSequence[sentenceIndex][wordIndex].tag
                if goldTag not in self.tagSet:
                    self.tagSet.add(goldTag)
                testTag = self.testTagSequence[sentenceIndex][wordIndex].tag
                if testTag not in self.tagSet:
                    self.tagSet.add(testTag)

        self.tagList = list(self.tagSet) # ordered
        for tag in self.tagList:
            self.confusionMatrix[tag] = {}
            for tag2 in self.tagList:
                self.confusionMatrix[tag][tag2] = 0

        # build confusion frequency matrix
        for sentenceIndex in range(len(self.goldTagSequence)):
            for wordIndex in range(len(self.goldTagSequence[sentenceIndex])):
                goldTag = self.goldTagSequence[sentenceIndex][wordIndex].tag
                testTag = self.testTagSequence[sentenceIndex][wordIndex].tag
                self.confusionMatrix[goldTag][testTag] += 1

    ################################
    #intput: None                  #
    #output: float                 #
    ################################
    def getTokenAccuracy(self):
        numberCorrect = 0
        numberTotal = 0
        for sentence in range(len(self.goldTagSequence)):
            for word in range(len(self.goldTagSequence[sentence])):
                if self.goldTagSequence[sentence][word].tag == self.testTagSequence[sentence][word].tag:
                    numberCorrect += 1
                numberTotal += 1
        retval = float(numberCorrect) / float(numberTotal)
        return retval

    ################################
    #intput: None                  #
    #output: float                 #
    ################################
    def getSentenceAccuracy(self):
        numberCorrect = 0
        numberTotal = 0
        for sentence in range(len(self.goldTagSequence)):
            numberTotal += 1
            correct = True
            for word in range(len(self.goldTagSequence[sentence])):
                if self.goldTagSequence[sentence][word].tag != self.testTagSequence[sentence][word].tag:
                    correct = False
                    break
            if correct:
                numberCorrect += 1
        retval = float(numberCorrect) / float(numberTotal)
        return retval

    ################################
    #intput:                       #
    #    outFile: string           #
    #output: None                  #
    ################################
    def writeConfusionMatrix(self, outFile):
        with open(outFile, 'w') as output:
            output.write(",".join(self.tagList) + "\n")
            for tag in self.tagList:
                output.write(tag+',')
                output.write(','.join([str(self.confusionMatrix[tag][tag2]) for tag2 in self.tagList])+'\n')

    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float                 #
    ################################
    def getPrecision(self, tagTi):
        testTags = 0
        testEqualsGoldTags = self.confusionMatrix[tagTi][tagTi]
        for tag in self.tagList:
            testTags += self.confusionMatrix[tag][tagTi]
        retval = float(testEqualsGoldTags) / float(testTags)
        return retval

    ################################
    #intput:                       #
    #    tagTi: string             #
    #output: float                 #
    ################################
    # Return the tagger's recall on gold tag t_j
    def getRecall(self, tagTj):
        goldTags = 0
        testEqualsGoldTags = self.confusionMatrix[tagTj][tagTj]
        for tag in self.tagList:
            goldTags += self.confusionMatrix[tagTj][tag]
        retval = float(testEqualsGoldTags) / float(goldTags)
        return retval


if __name__ == "__main__":
    # Pass in the gold and test POS-tagged data as arguments
    gold = "gold.txt"
    test = "out.txt"
    # You need to implement the evaluation class
    eval = Eval(gold, test)
    # Calculate accuracy (sentence and token level)
    print("Token accuracy: ", eval.getTokenAccuracy())
    print("Sentence accuracy: ", eval.getSentenceAccuracy())
    # Calculate recall and precision
    print("Recall on tag NNP: ", eval.getPrecision('NNP'))
    print("Precision for tag NNP: ", eval.getRecall('NNP'))
    # Write a confusion matrix
    eval.writeConfusionMatrix("confusion_matrix.txt")
