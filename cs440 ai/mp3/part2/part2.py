#!/usr/bin/env python2.7

import argparse
from math import log
import matplotlib.pyplot as plt
import itertools
import time

LAPLACE_SMOOTHING_CONSTANT = 0.1

def main():
    args = parse_args()
    (trainingLabels, trainingData) = parse_data(args.training_data)
    (testLabels, testData) = parse_data(args.test_data)

    # Sanity check to ensure labels and training data set are the same size
    assert(len(trainingLabels) == len(trainingData))
    assert(len(testLabels) == len(testData))

    # Init to bernoulli unless multinomial is specified
    calculate_likelihoods = calculate_likelihoods_bernoulli
    classifier = classify

    if args.classifier == 'MULTINOMIAL':
        calculate_likelihoods = calculate_likelihoods_multinomial
        # classifier = classify_multinomial
    
    classes = set(trainingLabels)

    print "Extracting vocabulary..."
    vocab = get_vocab(trainingData)

    print "Performing model for {} classification".format(args.classifier)

    print "Calculating likelihoods..."
    likelihoods = calculate_likelihoods(trainingData, trainingLabels, vocab)

    print "Calculating priors..."
    priors = calculate_priors(trainingLabels)

    print "Classifying test data..."
    classified = classifier(priors, likelihoods, testData, classes)

    print "Evaluating performance..."
    result = evaluate(classified, testLabels)

    print ""
    print "Results:"
    print "-----------"
    print "Overall accuracy: {:.2f}%".format(result)

    print ""
    print "Confusion matrix:"
    print_cm(confusion_matrix(classified, testLabels), classes)

    if not args.no_top10:
        print ""
        print "Top 10 words by likelihoods:"
        top_ll = top10_likelihoods(likelihoods, vocab, classes)
        for cls in top_ll:
            print "\tClass: {}".format(cls)
            for word in top_ll[cls]:
                print "\t\t{}".format(word) 

        print ""
        print "Top 10 words by odds ratio:"
        for word in top10_odds_ratio(likelihoods, vocab, classes):
            print "\t{}".format(word) 

def top10_likelihoods(likelihoods, vocab, classes):
    """ Gather the top 10 words by highest (descending) likelihoods for each class """
    resultDict = {}
    for cls in classes:
        results = []
        for word in vocab:
            results.append((word, likelihoods[cls][word]))
        resultDict[cls] = results
    # Sort and return top 10 for each class
    for key in resultDict:
        results = resultDict[key]
        resultDict[key] = map(lambda x: x[0], sorted(results, key=lambda x: x[1], reverse=True))[:10]
    return resultDict

def top10_odds_ratio(likelihoods, vocab, classes):
    """ Gather the top 10 words by highest (descending) odds ratios """
    results = []
    for word in vocab:
        highestOddsRatio = None
        for c1 in classes:
            for c2 in classes:
                # Skip self TODO: Is this right?
                # if c1 == c2:
                #     continue
                oddsRatio = odds_ratio(likelihoods, c1, c2, word)
                if oddsRatio > highestOddsRatio or highestOddsRatio == None:
                    highestOddsRatio = oddsRatio
        results.append((word, highestOddsRatio))
    # Sort and return top 10
    return map(lambda x: x[0], sorted(results, key=lambda x: x[1], reverse=True))[:10]

def odds_ratio(likelihoods, c1, c2, value):
    """ Calculate the odds ratio:
        odds(F = 1, c1, c2) = P(F = 1 | c1) / P(F = 1 | c2)
    """
    return log(float(likelihoods[c1][value]) / float(likelihoods[c2][value]))

def print_cm(cm, classes):
    ordered = sorted(classes)
    for row in ordered:
        for col in ordered:
            print "%.2f " % cm[row][col],
        print ""

def confusion_matrix(predicted, actual):
    classes = sorted(set(actual))
    matrix = {}
    total = len(actual)
    assert(total == len(predicted))
    fraction = 1.0 / float(total)
    for rowClass in classes:
        matrix[rowClass] = {}
        row = matrix[rowClass]
        actualIndices = set([x for x in range(total) if actual[x] == rowClass])
        actualNum = float(len(actualIndices))
        tests = [predicted[x] for x in range(total) if x in actualIndices]
        for confused in classes:
            classifiedCount = float(len(filter(lambda x: x == confused, tests)))
            row[confused] = classifiedCount / actualNum
    for x in matrix:
        for y in matrix[x]:
            matrix[x][y] *= 100.0
    return matrix

def evaluate(predicted, actual):
    """ Calculate the accuracy of our predicated vs. actual """
    assert(len(predicted) == len(actual))
    total = len(actual)
    correct = len([x for x in range(total) if predicted[x] == actual[x]])
    return (float(correct) / float(total)) * 100.0

def classify_multinomial(priors, likelihoods, testData, classes):
    """ TODO
    """
    print likelihoods
    sys.exit(0)

def classify(priors, likelihoods, testData, classes):
    """ Classify data based on bernoulli model
    """
    results = []
    for document in testData:
        bestClass = None
        bestProb = None
        currentProb = 0.0
        for cls in classes:
            prior = priors[cls]
            currentProb = log(prior)
            lhoods = likelihoods[cls]
            for (word, count) in document:
                if word in lhoods:
                    currentProb += log(lhoods[word])
                else:
                    currentProb += log(lhoods[None])
            if currentProb > bestProb or bestClass == None:
                bestProb = currentProb
                bestClass = cls
        results.append(bestClass)
    return results

def calculate_priors(trainingLabels):
    """ Estimate the priors for a class
    """
    sum = 0
    priors = {}
    totalSamples = len(trainingLabels)
    classes = set(trainingLabels)
    for cls in classes:
        numCls = len(filter(lambda x: x == cls, trainingLabels))
        sum += numCls
        priors[cls] = float(numCls) / float(totalSamples)
    
    # Sanity check: valid partitioning
    assert(sum == totalSamples)

    return priors

def calculate_likelihoods_multinomial(data, labels, vocab):
    """ Calculate the likelihoods for multinomial

    Likelihood format: likelihoods[CLASS][word]
    """
    likelihoods = {}
    counts = {}
    words = {}
    classes = set(labels)
    vocabLen = len(vocab)
    for cls in classes:
        # Initialize
        counts[cls] = {}
        words[cls] = 0
    # Perform counts
    line = 0
    for doc in data:
        cls = labels[line]
        wordCounts = counts[cls]
        for (word, count) in doc:
            if word not in wordCounts:
                wordCounts[word] = 0
            wordCounts[word] += count
            words[cls] += count
        line += 1
    # Compute likliehoods
    for cls in counts:
        wordCounts = counts[cls]
        likelihoods[cls] = {}
        wordsInClass = words[cls]
        for word in wordCounts:
            likelihoods[cls][word] = laplace_smooth(wordCounts[word], wordsInClass, vocabLen)
        # Add all training words:
        for word in vocab:
            if word not in likelihoods[cls]:
                likelihoods[cls][word] = laplace_smooth(0, wordsInClass, vocabLen)
        # Special laplace smoothing for words not found in training data
        likelihoods[cls][None] = laplace_smooth(0, wordsInClass, vocabLen)
    return likelihoods

def calculate_likelihoods_bernoulli(data, labels, vocab):
    """ Calculate the likelihoods for bernoulli

    Likelihood format: likelihoods[CLASS][word]
    """
    classes = set(labels)
    likelihoods = {}
    # Calculate likelihood for each class
    for cls in classes:
        documentsInClass = [set(map(lambda y: y[0], data[x])) for x in range(len(data)) if labels[x] == cls]
        numDocsInClass = len(documentsInClass)
        results = {}
        for word in vocab:
            numDocsWithWordInClass = len(filter(lambda x: word in x, documentsInClass))
            # Binary variable-- either present or not present
            results[word] = laplace_smooth(numDocsWithWordInClass, numDocsInClass, 2)
        # Special laplace smoothing for words not found in training data
        results[None] = laplace_smooth(0, numDocsInClass, 2)
        likelihoods[cls] = results
    return likelihoods

def get_vocab(trainingData):
    """ Extract the known vocabulary from our training data """
    return set(reduce(lambda x,y: x+y, map(lambda x: map(lambda y: y[0], x), trainingData), []))

def laplace_smooth(num, denom, numVals):
    k = float(LAPLACE_SMOOTHING_CONSTANT)
    return (float(num) + k) / (float(denom) + (float(numVals) * k))

def parse_data(filename):
    """ Parse and explode input data """
    labels = []
    documents = []
    with open(filename, 'r') as f:
        for line in f:
            values = line.split()
            label = values[0]
            document = []
            for wordCount in values[1:]:
                parsed = wordCount.split(':')
                word = parsed[0]
                count = int(parsed[1])
                document.append((word, count))
            labels.append(label)
            documents.append(document)
    return (labels, documents)

def parse_args():
    parser = argparse.ArgumentParser(description="Classify digits")
    parser.add_argument("-t", "--training-data", required=True)
    parser.add_argument("-d", "--test-data", required=True)
    parser.add_argument("-c", "--classifier", choices=['BERNOULLI', 'MULTINOMIAL'], required=True)
    parser.add_argument("-n", "--no-top10", action="store_true")
    parser.set_defaults(no_top10=False)
    return parser.parse_args()

if __name__ == "__main__":
    main()
