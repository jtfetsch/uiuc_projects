#!/usr/bin/env python2.7
"""
part2.py
"""

import random
import argparse
import heapq
import math
import pickle

BIAS = 0.1
ALPHA = 0.65
RAND_BOUNDS = (-1.0, 1.0)

def main():
    global BIAS
    args = parse_args()
    train = parse_images(args.train)
    trainLabels = parse_labels(args.train_labels)
    test = parse_images(args.test)
    testLabels = parse_labels(args.test_labels)
    BIAS = args.bias

    assert len(train) == len(trainLabels) and len(test) == len(testLabels)

    print "Successfully parsed {} training images and {} test images.".format(len(train), len(test))

    classified = None

    if not args.nearest_neighbor:
        # Neural network
        print "Initializing weight vector..."
        initial = initialize_weights(args.randomize_weights)

        print "Training network..." 
        (weights, trainingAccuracy) = train_model(sgn, initial, train, trainLabels, args.epochs, args.randomize_epoch)

        print "Testing network..."
        classified = test_model(sgn, weights, test)

        print "Training accuracy: {}".format(trainingAccuracy)
    else:
        # Nearest neighbor classification
        k = args.nn_k
        numSamples = args.num_train_samples if args.num_train_samples else len(train)
        classified = classify_knn(k, train[:int(numSamples)], trainLabels[:int(numSamples)], test, manhattan)
    
    (overallAcc, perDigitAcc) = compare_accuracy(classified, testLabels)

    print "Overall accuracy (on test set): {}".format(overallAcc)

    print_cm(confusion_matrix(classified, testLabels), range(10))

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

def manhattan(im1, im2, i, j):
    return abs(im1[i][j] - im2[i][j])

def sgn(v):
    return -1.0 if v < 0.0 else (0.0 if v == 0.0 else 1.0)

def classify_knn(k, train, labels, test, distance):
    assert len(train) == len(labels)
    classifications = []
    classified = 0
    for image in test:
        if classified % 100 == 0:
            print "Classified {}/{} images".format(classified, len(test))
        results = [(labels[i], im_dist(image, train[i], distance)) for i in range(len(labels))]
        results.sort(key=lambda x: x[1])
        cohort = [(x[0], results.count(x[0])) for x in results[:int(k)]]
        cohort.sort(key=lambda x: x[1])
        classifications.append(cohort[0][0])
        classified += 1
    return classifications

def im_dist(im1, im2, distance):
    totalDist = 0.0
    for i in range(28):
        for j in range(28):
            totalDist += distance(im1, im2, i, j)
    return totalDist

def compare_accuracy(classified, testLabels):
    assert len(classified) == len(testLabels)
    digitSuccess = [0.0] * 10
    digitCount = [0.0] * 10
    overall = 0.0
    for i in range(len(classified)):
        actual = classified[i]
        expected = testLabels[i]
        if actual == expected:
            digitSuccess[expected] += 1.0
            overall += 1.0
        digitCount[expected] += 1.0
    overall /= float(len(classified))
    perDigit = [digitSuccess[i] / digitCount[i] for i in range(10)]
    return (overall, perDigit)

def test_model(activationFn, weights, data):
    classifications = []
    for image in data:
        classification = None
        bestVal = None
        for cls in range(10):
            rowNum = 0
            output = BIAS
            for row in image:
                colNum = 0
                for col in row:
                    output += col * weights[cls][rowNum][colNum]
                    colNum += 1
                rowNum += 1
            output = activationFn(output)
            if output > bestVal:
                bestVal = output
                classification = cls
        classifications.append(classification)
    return classifications

def train_model(activationFn, initial, data, labels, epochs, random_epoch):
    """
    Train the multi-class perceptron
    """
    global ALPHA
    assert len(data) == len(labels)
    weights = list(initial)
    imageData = list(data)
    trainingSucceed = 0.0
    trainingCount = 0.0
    for T in range(epochs):
        print "Training on epoch {}/{}".format(T + 1, epochs)
        if random_epoch:
            random.shuffle(imageData)
        imageNum = 0
        for image in imageData:
            classified = None
            bestVal = None
            # Take argmax over all classes and try to classifiy
            for cls in range(10):
                output = BIAS
                rowNum = 0
                for row in image:
                    colNum = 0
                    for col in row:
                        output += col * weights[cls][rowNum][colNum]
                        colNum += 1
                    rowNum += 1
                output = activationFn(output)
                if output > bestVal:
                    bestVal = output
                    classified = cls
            # Update weights if misclassified
            wanted = labels[imageNum]
            if classified != wanted:
                wrong = classified
                rowNum = 0
                for row in image:
                    colNum = 0
                    for col in row:
                        update = ALPHA * col
                        weights[wanted][rowNum][colNum] += update
                        weights[wrong][rowNum][colNum] -= update
                        colNum += 1
                    rowNum += 1
            else:
                trainingSucceed += 1.0
            trainingCount += 1.0
            imageNum += 1
        # Decay alpha each epoch
        #ALPHA = float(50.0 / (50.0 + float(T) + 1.0)) * ALPHA
    return (weights, (trainingSucceed / trainingCount))

def initialize_weights(randomize_weights):
    """ Initialize weight vector. Indexed by:
            weight[CLASS][ROW][COL]
    """
    weights = []
    for cls in range(10): # Weight vector per class
        clsWeights = []
        for row in range(28):
            rowWeights = []
            for col in range(28):
                # Weight for each pixel
                rowWeights.append(0.0 if not randomize_weights else float(random.randint(*RAND_BOUNDS)))
            clsWeights.append(rowWeights)
        weights.append(clsWeights)
    return weights

def parse_images(imageFile):
    """ Parse image files. Files _must_ represent images of 28x28 pixels
        NOTE: Logic directly taken from MP3 part 1 parsing code
    """
    images = []
    with open(imageFile, 'r') as f:
        currentImage = []
        count = 0 
        for row in f:
            # Sanity check for rows
            assert(len(row) >= 28) 
            if len(row) > 28: 
                # Ensure only trailing whitespace for any overlap
                assert(len(row[28:].strip()) == 0)
            # After other sanity checks, just take first 28 chars
            assert(len(row[:28]) == 28) 
            # 0 for background and 1 for foreground since we don't make a distinction
            # between the foreground types (as per MP spec)
            convertedRow = map(lambda x: 0 if x == ' ' else 1, row[:28])
            currentImage.append(convertedRow)
            count += 1
            # Added 28 rows, need to start next image
            if count % 28 == 0:
                assert(len(currentImage) == 28)
                images.append(currentImage)
                currentImage = []
                count = 0
    return images

def parse_labels(labelFile):
    """ Parse the labels. These are assumed to be integers (one per line)
        NOTE: Logic directly taken from MP3 part 1 parsing code
    """
    with open(labelFile, 'r') as f:
        return [int(x) for x in f]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", help="Training data", required=True)
    parser.add_argument("-l", "--train-labels", help="Training data labels", required=True)
    parser.add_argument("-d", "--test", help="Test data", required=True)
    parser.add_argument("-L", "--test-labels", help="Test data labels", required=True)
    parser.add_argument("-r", "--randomize-weights", help="Randomize initial weight vector", action='store_true')
    parser.add_argument("-e", "--epochs", help="Number of epochs", default=5, type=int)
    parser.add_argument("-b", "--bias", help="Bias value to apply", default=0.1, type=float)
    parser.add_argument("-a", "--alpha", help="Learning rate parameter", default=0.65, type=float)
    parser.add_argument("-o", "--randomize-epoch", help="Randomize images in each epoch", action='store_true')
    parser.add_argument("-n", "--nearest-neighbor", help="Run a nearest neighbor classification", action='store_true')
    parser.add_argument("-k", "--nn-k", help="k value for KNN", default=5)
    parser.add_argument("-s", "--num-train-samples", help="Number of training samples to use in KNN classifier", type=int)
    return parser.parse_args()

if __name__ == "__main__":
    main()
