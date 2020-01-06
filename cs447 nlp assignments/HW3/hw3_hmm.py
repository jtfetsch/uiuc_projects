##
## Part 1:
## Train a bigram HMM for POS tagging
##
import os.path
import sys
import numpy
from operator import itemgetter
from collections import defaultdict
from math import log

# Unknown word token
UNK = 'UNK'
LOG_ZERO = -2**32

# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_');
        self.word = parts[0]
        self.tag = parts[1]

def safe_log(value):
    if value == 0:
        return LOG_ZERO
    return log(value, 2)


# Class definition for a bigram HMM
class HMM:
### Helper file I/O methods ###
    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads a labeled data inputFile, and returns a nested list of sentences, where each sentence is a list of TaggedWord objects
    def readLabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r")  # open the input file in read-only mode
            sens = [];
            for line in file:
                raw = line.split()
                sentence = []
                for token in raw:
                    sentence.append(TaggedWord(token))
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script

    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads an unlabeled data inputFile, and returns a nested list of sentences, where each sentence is a list of strings
    def readUnlabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = [];
            for line in file:
                sentence = line.split() # split the line into a list of words
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s ddoes not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script
### End file I/O methods ###

    ################################
    #intput:                       #
    #    unknownWordThreshold: int #
    #output: None                  #
    ################################
    # Constructor
    def __init__(self, unknownWordThreshold=5):
        # Unknown word threshold, default value is 5 (words occurring fewer than 5 times should be treated as UNK)
        self.minFreq = unknownWordThreshold
        self.tagCounts = {}     # raw counts of tags in train       stored as [tag] = count
        self.wordCounts = {}    # raw counts of words in train      stored as [word] = count
        self.wordTags = {}      # enumerate tags that a word has    stored as [word][tag] = count
        self.transitions = {}   # p(tag | previous_tag)             stored as [previous_tag][tag] = count
        self.emissions = {}     # p(word | tag)                     stored as [tag][word] = count
        self.initials = {}      # p(tag at start of sentence)       stored as [tag] = count
        self.numWords = 0       # num_words
        self.sentenceCount = 0  # num_sentences

    ################################
    #intput:                       #
    #    trainFile: string         #
    #output: None                  #
    ################################
    # Given labeled corpus in trainFile, build the HMM distributions from the observed counts
    def train(self, trainFile):
        data = self.readLabeledData(trainFile)  # data is a nested list of TaggedWords
        # iterate a first time: do UNK evaluation in wordCounts[word] => frequency, record initial tag probabilities
        for line in data:
            self.sentenceCount += 1
            if line[0].tag not in self.initials:
                self.initials[line[0].tag] = 0
            self.initials[line[0].tag] += 1
            for tagword in line:
                word = tagword.word
                if tagword.tag not in self.tagCounts:
                    self.tagCounts[tagword.tag] = 0
                self.tagCounts[tagword.tag] += 1
                if word not in self.wordCounts:
                    self.wordCounts[word] = 0
                self.wordCounts[word] += 1
                self.numWords += 1

        # iterate a second time: calculate emission probabilities, word tag counts with UNK token
        for line in data:
            for tagword in line:
                # emissions
                word = tagword.word if self.wordCounts[tagword.word] >= self.minFreq else "UNK"
                tag = tagword.tag
                if tag not in self.emissions:
                    self.emissions[tag] = {}  # word => count
                if word not in self.emissions[tag]:
                    self.emissions[tag][word] = 0
                self.emissions[tag][word] += 1
                # word tags
                if word not in self.wordTags:
                    self.wordTags[word] = {}
                if tag not in self.wordTags[word]:
                    self.wordTags[word][tag] = 0
                self.wordTags[word][tag] += 1

        # iterate a third time: calculate transition probabilities
        for line in data:
            for tagIndex in range(1, len(line)):
                tag = line[tagIndex].tag
                previousTag = line[tagIndex-1].tag
                if previousTag not in self.transitions:
                    self.transitions[previousTag] = {}
                if tag not in self.transitions[previousTag]:
                    self.transitions[previousTag][tag] = 0
                self.transitions[previousTag][tag] += 1

        # do Laplace smoothing for transition probabilities here
        for tag in self.tagCounts:
            if tag not in self.transitions:
                self.transitions[tag] = {}
            for tag2 in self.tagCounts:
                if tag2 not in self.transitions[tag]:
                    self.transitions[tag][tag2] = 0
                self.transitions[tag][tag2] += 1

    ################################
    #intput:                       #
    #     testFile: string         #
    #    outFile: string           #
    #output: None                  #
    ################################
    # Given an unlabeled corpus in testFile, output the Viterbi tag sequences as a labeled corpus in outFile
    def test(self, testFile, outFile):
        data = self.readUnlabeledData(testFile)
        f=open(outFile, 'w+')
        for sen in data:
            vitTags = self.viterbi(sen)
            senString = ''
            for i in range(len(sen)):
                senString += sen[i]+"_"+vitTags[i]+" "
            print(senString.rstrip(), end="\n", file=f)

    ################################
    #intput:                       #
    #    words: list               #
    #output: list                  #
    ################################
    # Given a list of words, runs the Viterbi algorithm and returns a list containing the sequence of tags
    # that generates the word sequence with highest probability, according to this HMM
    def viterbi(self, words):
        # do word count filtering on words
        UNK_filtered_words = []
        for word in words:
            lower_word = word
            if lower_word in self.wordCounts:
                if self.wordCounts[lower_word] >= self.minFreq:
                    UNK_filtered_words.append(lower_word)
                    continue
            UNK_filtered_words.append("UNK")
        #print(UNK_filtered_words)

        # build grid
        viterbiConstruct = [] # [word][tag] = (log_prob, back_ptr)
        tags = list(self.tagCounts.keys())
        for i in range(len(UNK_filtered_words)):
            viterbiConstruct.append([(LOG_ZERO, -1)] * len(tags))

        # initialize first values
        for tag in range(len(tags)):
            pi_tk = self.get_initial_tag_probability(tags[tag])
            p_w_tk = self.get_emission_probability(UNK_filtered_words[0], tags[tag])
            viterbiConstruct[0][tag] = (safe_log(pi_tk * p_w_tk), -1)

        # calculate values for rest of viterbi grid [wordI][tagJ] from [wordI-1][tagK]
        for wordIndexI in range(1, len(UNK_filtered_words)):
            wordI = UNK_filtered_words[wordIndexI]
            #print("evaluating p() for word", wordI, "with tags:", self.wordTags[wordI])
            for tagIndexJ in range(len(tags)):
                tagJ = tags[tagIndexJ]
                max_value = LOG_ZERO
                max_index = -1
                emission_prob = self.get_emission_probability(wordI, tagJ)
                # if a word has zero probability for a specific tag tagJ, we don't need to compute further
                if emission_prob == 0:
                    viterbiConstruct[wordIndexI][tagIndexJ] = (LOG_ZERO, -1)
                    continue

                #print("evaluating potential tag", tagJ)
                for tagIndexK in range(len(tags)):
                    tagK = tags[tagIndexK]
                    # if a previous state has zero probability, we don't need to consider it
                    if viterbiConstruct[wordIndexI-1][tagIndexK][0] == LOG_ZERO:
                        continue

                    value = viterbiConstruct[wordIndexI-1][tagIndexK][0] + \
                            safe_log(self.get_transition_probability(tagK, tagJ) * emission_prob)
                    if value > max_value:
                        max_value = value
                        max_index = tagIndexK
                #print("max score =", max_value, "from", tags[max_index], "to", tagJ, "on word:", wordI)
                #print("\tprevious viterbi:", viterbiConstruct[wordIndexI-1][max_index])
                #print("\ttransition p(%s|%s):" %(tagJ, tags[max_index]), safe_log(self.get_transition_probability(tags[max_index], tagJ)))
                #print("\temission p(%s|%s):" %(wordI, tagJ), safe_log(emission_prob))
                viterbiConstruct[wordIndexI][tagIndexJ] = (max_value, max_index)

        sentenceTags = []
        maxValue = LOG_ZERO
        nextIndex = -1
        for tagIndex in range(len(tags)):
            if viterbiConstruct[-1][tagIndex][0] > maxValue:
                maxValue = viterbiConstruct[-1][tagIndex][0]
                nextIndex = tagIndex

        sentenceTags.append(tags[nextIndex])
        for wordIndex in range(len(UNK_filtered_words)-2, -1, -1):
            nextIndex = viterbiConstruct[wordIndex+1][nextIndex][1]
            sentenceTags.append(tags[nextIndex])

        sentenceTags.reverse()

        #print(sentenceTags)
        return sentenceTags

    def get_initial_tag_probability(self, tag):
        if tag not in self.initials:
            return 0
        return float(self.initials[tag]) / float(self.sentenceCount)

    def get_emission_probability(self, word, tag):
        if tag not in self.tagCounts:
            return 0
        if tag not in self.emissions:
            return 0
        if word not in self.emissions[tag]:
            return 0
        return float(self.emissions[tag][word]) / float(self.tagCounts[tag])

    def get_transition_probability(self, previous_tag, current_tag):
        # Laplace smoothing on transitions
        if previous_tag not in self.transitions:
            return 0
        if current_tag not in self.transitions[previous_tag]:
            return 0
        totalTransitions = 0
        for tag in self.transitions[previous_tag]:
            totalTransitions += self.transitions[previous_tag][tag]
        return float(self.transitions[previous_tag][current_tag]) / float(totalTransitions)


if __name__ == "__main__":
    tagger = HMM()
    tagger.train('train.txt')
    tagger.test('test.txt', 'out.txt')
