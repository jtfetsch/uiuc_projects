##
## Part 1:
## Develop a smoothed n-gram language model and evaluate it on a corpus
##
import os.path
import sys
import random
import math
from operator import itemgetter
from collections import defaultdict
#----------------------------------------
#  Data input 
#----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        print("Reading file ", f)
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            #append this lis as an element to the list of sentences
            corpus.append(sentence)
            if i % 1000 == 0:
    	#print a status message: str(i) turns int i into a string
    	#so we can concatenate it
                sys.stderr.write("Reading sentence " + str(i) + "\n")
        #endif
    #endfor
        return corpus
    else:
    #ideally we would throw an exception here, but this will suffice
        print("Error: corpus file ", f, " does not exist")
        sys.exit() # exit the script
    #endif
#enddef


# Preprocess the corpus to help avoid sparsity
def preprocess(corpus):
    #find all the rare words
    freqDict = defaultdict(int)
    for sen in corpus:
	    for word in sen:
	       freqDict[word] += 1
	#endfor
    #endfor

    #replace rare words with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if freqDict[word] < 2:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor

    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor
    
    return corpus
#enddef

def preprocessTest(vocab, corpus):
    #replace test words that were unseen in the training with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if word not in vocab:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor
    
    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor

    return corpus
#enddef

# Constants 
UNK = "UNK"     # Unknown word token
start = "<s>"   # Start-of-sentence token
end = "</s>"    # End-of-sentence-token


#--------------------------------------------------------------
# Language models and data structures
#--------------------------------------------------------------

# Parent class for the three language models you need to implement
class LanguageModel:
    # Initialize and train the model (ie, estimate the model's underlying probability
    # distribution from the training corpus)
    def __init__(self, corpus):
        print("""Your task is to implement five kinds of n-gram language models:
      a) an (unsmoothed) unigram model (UnigramModel)
      b) a unigram model smoothed using Laplace smoothing (SmoothedUnigramModel)
      c) an unsmoothed bigram model (BigramModel)
      """)
    #enddef

    # Generate a sentence by drawing words according to the 
    # model's probability distribution
    # Note: think about how to set the length of the sentence 
    #in a principled way
    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        return "mary had a little lamb ."
    #emddef

    # Given a sentence (sen), return the probability of 
    # that sentence under the model
    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")
        return 0.0
    #enddef

    # Given a corpus, calculate and return its perplexity 
    #(normalized inverse log probability)
    def getCorpusPerplexity(self, corpus):
        print("Implement the getCorpusPerplexity method")
        return 0.0
    #enddef

    # Given a file (filename) and the number of sentences, generate a list
    # of sentences and write each to file along with its model probability.
    # Note: you shouldn't need to change this method
    def generateSentencesToFile(self, numberOfSentences, filename):
        filePointer = open(filename, 'w+')
        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)
            stringGenerated = str(prob) + " " + " ".join(sen) 
            
	#endfor
    #enddef
#endclass

# Unigram language model
class UnigramModel(LanguageModel):
    def __init__(self, corpus):
        self.UnigramDis = UnigramDist(corpus)
    #endddef

    def generateSentence(self):
        sentence = [start]
        while True:
            word = self.UnigramDis.draw()
            sentence.append(word)
            if word == end:
                break
        return sentence
    #enddef

    def getSentenceProbability(self, sen):
        prob = 1.0
        for word in sen[1:]:
            prob *= self.UnigramDis.prob(word)
        return prob
    #enddef

    def getCorpusPerplexity(self, corpus):
        logSum = 0.0
        numWords = 0.0
        for sen in corpus:
            for word in sen[1:]:
                numWords += 1.0
                evalWord = UNK
                if word in self.UnigramDis.counts.keys():
                    evalWord = word
                if self.UnigramDis.prob(evalWord) == 0:
                    return math.inf
                logSum += math.log(self.UnigramDis.prob(evalWord))
        perp = math.exp(-1.0/numWords * logSum)
        return perp
    #enddef
#endclass

#Smoothed unigram language model (use laplace for smoothing)
class SmoothedUnigramModel(LanguageModel):
    def __init__(self, corpus):
        self.UnigramSmoothedDis = UnigramSmoothedDist(corpus)
    #endddef

    def generateSentence(self):
        sentence = [start]
        while True:
            word = self.UnigramSmoothedDis.draw()
            sentence.append(word)
            if word == end:
                break
        return sentence
    #enddef

    def getSentenceProbability(self, sen):
        prob = 1.0
        for word in sen[1:]:
            prob *= self.UnigramSmoothedDis.prob(word)
        return prob
    #enddef

    def getCorpusPerplexity(self, corpus):
        logSum = 0.0
        numWords = 0.0
        for sen in corpus:
            for word in sen[1:]:
                evalWord = UNK
                numWords += 1.0
                if word in self.UnigramSmoothedDis.counts.keys():
                    evalWord = word
                if self.UnigramSmoothedDis.prob(evalWord) == 0:
                    return math.inf
                logSum += math.log(self.UnigramSmoothedDis.prob(evalWord))
        perp = math.exp(-1.0/numWords * logSum)
        return perp
    #enddef
#endclass

# Unsmoothed bigram language model
class BigramModel(LanguageModel):
    def __init__(self, corpus):
        self.BigramDis = BigramDist(corpus)
    #endddef

    def generateSentence(self):
        sentence = [start]
        while True:
            word = self.BigramDis.draw(sentence[-1])
            sentence.append(word)
            if word == end:
                break
        return sentence
    #enddef

    def getSentenceProbability(self, sen):
        prob = 1.0
        for index in range(len(sen)-1):
            prob *= self.BigramDis.prob(sen[index], sen[index+1])
        return prob
    #enddef

    def getCorpusPerplexity(self, corpus):
        logSum = 0.0
        numWords = 0.0
        for sen in corpus:
            for index in range(len(sen)-1):
                numWords += 1.0
                probValue = self.BigramDis.prob(sen[index], sen[index+1])
                if probValue != 0.0:
                    logSum += math.log(probValue)
                else:
                    return math.inf
        return math.exp(-1.0/numWords * logSum)
    #enddef
    #endddef
#endclass

class UnigramDist:
    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)
    #endddef

    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0
            #endfor
        #endfor
    #enddef

    # Returns the probability of word in the distribution
    def prob(self, word):
        return self.counts[word]/self.total
    #enddef

    # Generate a single random word according to the distribution
    def draw(self):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.prob(word)
            if rand <= 0.0:
                return word
	    #endif
	#endfor
    #enddef
#endclass

class UnigramSmoothedDist(UnigramDist):
    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0
            #endfor
        #endfor

        self.unsmoothedTotal = self.total

        # smoothing done here
        for word in self.counts:
            self.counts[word] += 1.0
            self.total += 1.0
    #enddef
#endclass

class BigramDist:
    def __init__(self, corpus):
        self.counts = {}
        self.total = 0.0
        self.train(corpus)
    #endddef

    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        for sen in corpus:
            for index in range(1,len(sen)):
                if sen[index-1] not in self.counts:
                    self.counts[sen[index-1]] = defaultdict(float)
                self.counts[sen[index-1]][sen[index]]+=1.0
                self.total += 1.0
            #endfor
        #endfor
    #enddef

    # Returns the probability of word in the distribution
    def prob(self, word, word2):
        if word not in self.counts:
            return 0.0
        if word2 not in self.counts[word]:
            return 0.0
        total = 0.0
        for secondWord in self.counts[word]:
            total += self.counts[word][secondWord]
        return self.counts[word][word2]/total
    #enddef

    # Generate a single random word according to the distribution
    def draw(self, precedingWord):
        rand = random.random()
        for word in self.counts[precedingWord].keys():
            rand -= self.prob(precedingWord, word)
            if rand <= 0.0:
                return word
        #endif
    #endfor
    #enddef
#endclass

#-------------------------------------------
# The main routine
#-------------------------------------------
if __name__ == "__main__":
    #read your corpora
    trainCorpus = readFileToCorpus('train.txt')
    trainCorpus = preprocess(trainCorpus)
    
    posTestCorpus = readFileToCorpus('pos_test.txt')
    negTestCorpus = readFileToCorpus('neg_test.txt')
    
    vocab = set()
    for sen in trainCorpus:
        for word in sen:
            vocab.add(word)

    posTestCorpus = preprocessTest(vocab, posTestCorpus)
    negTestCorpus = preprocessTest(vocab, negTestCorpus)

    # Run sample unigram dist code
    unigramDist = UnigramDist(trainCorpus)
    print("Sample UnigramDist output:")
    print("Probability of \"vader\": ", unigramDist.prob("vader"))
    print("Probability of \""+UNK+"\": ", unigramDist.prob(UNK))
    print("\"Random\" draw: ", unigramDist.draw())