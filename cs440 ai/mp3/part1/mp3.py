#!/usr/bin/env python2.7

import argparse
from math import log
import matplotlib.pyplot as plt
import itertools
import time

LAPLACE_SMOOTHING_CONSTANT = 0.1
PIXEL_GROUP_ROWS = 1
PIXEL_GROUP_COLS = 1
PIXEL_GROUP_OVERLAP = True
LEAST_PROTOTYPICAL = {}
MOST_PROTOTYPICAL = {}

def main():
    global PIXEL_GROUP_ROWS, PIXEL_GROUP_COLS, PIXEL_GROUP_OVERLAP
    args = parse_args()
    trainingLabels = parse_labels(args.training_labels)
    trainingData = parse_images(args.training_data)
    testData = parse_images(args.test_data)
    testLabels = parse_labels(args.test_labels)
    PIXEL_GROUP_OVERLAP = args.overlap
    PIXEL_GROUP_COLS = args.group_cols
    PIXEL_GROUP_ROWS = args.group_rows

    # Sanity check to ensure labels and training data set are the same size
    assert(len(trainingLabels) == len(trainingData))

    print "Successfully parsed {} training labels and 28x28 images!".format(len(trainingLabels), len(trainingData))

    startTrain = time.time()
    print "Calculating likelihoods..."
    likelihoods = calculate_likelihoods(trainingData, trainingLabels)

    print "Estimating priors..."
    priors = calculate_priors(trainingLabels)
    endTrain = time.time()

    startClassify = time.time()
    print "Classifying test data..."
    classified = classify_MAP(priors, likelihoods, testData, set(trainingLabels))
    endClassify = time.time()

    print "Evaluating accuracy..."
    classes = sorted(set(trainingLabels))
    for cls in classes:
        digitAccuracy = evaluate_class(cls, classified, testLabels)
        print "\tDigit {}: {:.2f}%".format(cls, digitAccuracy)
    accuracy = evaluate(classified, testLabels)
    print "Overall Accuracy: {}%".format(accuracy)

    # Only compute for 1x1
    if PIXEL_GROUP_COLS == 1 and PIXEL_GROUP_ROWS == 1:
        print "Confusion matrix:"
        cm = confusion_matrix(classified, testLabels)
    
        print_cm(cm, classes)

        confusion_list = sorted([(x, y, cm[x][y]) for x in cm for y in cm[x] if x != y], key=lambda x: x[2], reverse=True)
        plot_confusions(likelihoods, confusion_list[:4])

        print "Reporting prototypes for each class:"
        for cls in classes:
            print "Most prototypical for: {}".format(cls)
            save_image(MOST_PROTOTYPICAL[cls][1], "most{}.txt".format(cls))
            print "Least prototypical for: {}".format(cls)
            save_image(LEAST_PROTOTYPICAL[cls][1], "least{}.txt".format(cls))
    
    print "Spent {:.2f}s training and {:.2f}s classifying".format((endTrain - startTrain), (endClassify - startClassify))

def save_image(image, filename):
    imageFile = 'prototypes/{}'.format(filename)
    with open(imageFile, 'w') as f:
        lines = []
        for row in image:
            line = ""
            for c in row:
                line += str(c)
            lines.append(line + '\n')
        f.writelines(lines)
    print 'Saved: {}'.format(imageFile)

def plot_confusions(likelihoods, confusions):
    likelihoodImage = lambda x,y: [log(likelihoods[x][pos_to_idx(i, j)][y]) for i in range(28) for j in range(28)]
    for (row, col, rate) in confusions:
        plot_image(likelihoodImage(row, "1"), "likelihood{}.png".format(row))
        plot_image(likelihoodImage(col, "1"), "likelihood{}.png".format(col))
        plot_image(odds_ratio(likelihoods, row, col, "1"), "odds{}_{}.png".format(row, col))

def plot_image(image, filename=None):
    """ Plot a 2D 28x28 image. """
    plot2d = []
    if type(image[0]) == list:
        # Already exploded as 2D
        plot2d = image
    else:
        # Convert
        for i in range(28):
            row = []
            for j in range(28):
                row.append(image[pos_to_idx(i, j)])
            plot2d.append(row)
    plt.clf()
    plt.imshow(plot2d, interpolation='nearest')
    plt.colorbar()
    if filename:
        print "Saved: plots/{}".format(filename)
        plt.savefig("plots/{}".format(filename))
    else:
        plt.show()

def odds_ratio(likelihoods, c1, c2, value):
    """ Calculate the odds ratio:
        odds(Fij = 1, c1, c2) = P(Fij = 1 | c1) / P(Fij = 1 | c2)
    """
    result = []
    for i in range(28):
        for j in range(28):
            idx = pos_to_idx(i, j)
            # NOTE: We take the log of our likelihood calculation for more useful
            # visualization (just like in the MP spec)
            calc = log(float(likelihoods[c1][idx][value]) / float(likelihoods[c2][idx][value]))
            result.append(calc)
    return result

def print_cm(cm, classes):
    ordered = sorted(classes)
    for row in ordered:
        for col in ordered:
            print "%.2f " % cm[row][col],
        print ""

def confusion_matrix(predicted, actual):
    classes = set(actual)
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

def evaluate_class(cls, predicted, testLabels):
    """ For a particular class, provide the accuracy """
    total = len(testLabels)
    assert(total == len(predicted))
    correct = 0
    testNum = 0
    for x in range(total):
        if testLabels[x] == cls:
            testNum += 1
            if predicted[x] == cls:
                correct += 1
    return (float(correct) / float(testNum)) * 100.0

def evaluate(predicted, actual):
    """ Calculate the accuracy of our predicated vs. actual """
    assert(len(predicted) == len(actual))
    total = len(actual)
    correct = len([x for x in range(total) if predicted[x] == actual[x]])
    return (float(correct) / float(total)) * 100.0

def classify_MAP(priors, likelihoods, testData, classes):
    """ Classify testData according to the precomputed priors and likelihoods.
    
        This is using the MAP algorithm, so the classification will be decided
        by the class with the highest probability.
    """
    classifications = []
    for image in testData:
        bestClass = None
        bestValue = None
        for cls in classes:
            value = log(priors[cls])
            for i in range(28):
                for j in range(28):
                    idx = pos_to_idx(i, j)
                    groupValue = get_group_value(image, i, j)
                    value += log(likelihoods[cls][idx][groupValue])
            if cls not in MOST_PROTOTYPICAL:
                MOST_PROTOTYPICAL[cls] = (value, image)
                LEAST_PROTOTYPICAL[cls] = (value, image)
            elif MOST_PROTOTYPICAL[cls][0] < value:
                MOST_PROTOTYPICAL[cls] = (value, image)
            elif LEAST_PROTOTYPICAL[cls][0] > value:
                LEAST_PROTOTYPICAL[cls] = (value, image)
            if bestValue == None or value > bestValue:
                bestValue = value
                bestClass = cls
        classifications.append(bestClass)
    return classifications                    

def calculate_priors(trainingLabels):
    """ Calculate priors for each class based on empirical frequency of class
        in training data
    """
    classes = set(trainingLabels)
    sum = 0
    priors = {}
    totalSamples = len(trainingLabels)

    for cls in classes:
        numCls = len(filter(lambda x: x == cls, trainingLabels))
        sum += numCls
        priors[cls] = float(numCls) / float(totalSamples)
    
    # Sanity check: valid partitioning
    assert(sum == totalSamples)

    return priors

def calculate_likelihoods(data, labels):
    """ Calculate the likelihoods for all classes and each pixel

        Likelihoods are returned as a list of dictionaries. The dictionary is keyed as:
          likelihood[CLASS][idx][VALUE] meaning P(Fidx = VALUE | CLASS)
    """
    classes = set(labels)
    likelihoods = {}
    # Calculate likelihood for each class
    for cls in classes:
        likelihoods[cls] = class_likelihood(find_images_in_class(data, labels, cls))
    return likelihoods

def class_likelihood(imagesInClass):
    counts = []
    likelihoods = []
    distinctKeys = ["".join(x) for x in itertools.product("01", repeat = PIXEL_GROUP_ROWS * PIXEL_GROUP_COLS)]
    # Initialize all counts to 0
    for i in range(28):
        for j in range(28):
            values = {}
            for x in distinctKeys:
                values[x] = 0
            counts.append(values)
    
    # Count for each image
    for image in imagesInClass:
        for i in range(28):
            for j in range(28):
                idx = pos_to_idx(i, j)
                groupValue = None
                if PIXEL_GROUP_OVERLAP or ((i % PIXEL_GROUP_ROWS == 0) and (j % PIXEL_GROUP_COLS == 0)):
                    # Overlapping subsets
                    groupValue = get_group_value(image, i, j)
                else:
                    # Non-overlapping (i.e. each subset retains the same value as group rep)
                    groupI = i - (i % PIXEL_GROUP_ROWS)
                    groupJ = j - (j % PIXEL_GROUP_COLS)
                    groupValue = get_group_value(image, groupI, groupJ)
                counts[idx][groupValue] += 1
    
    numImagesInClass = len(imagesInClass)
    
    # Calculate likelihood
    for count in counts:
        likelihoodValues = {}
        for key in count:
            likelihoodValues[key] = laplace_smooth(count[key], numImagesInClass, len(count))
        likelihoods.append(likelihoodValues)

    return likelihoods 

def get_group_value(image, i, j):
    groupValue = ""
    # Combine keys for pixel groups
    for x in range(PIXEL_GROUP_ROWS):
        groupI = (i + x) % 28
        for y in range(PIXEL_GROUP_COLS):
            groupJ = (j + y) % 28
            groupValue += str(image[groupI][groupJ])
    return groupValue

def pos_to_idx(i, j):
    return (i * 28) + j

def idx_to_pos(idx):
    j = idx % 28
    i = (idx - j) / 28
    return (i, j)

def laplace_smooth(num, denom, numVals):
    k = float(LAPLACE_SMOOTHING_CONSTANT)
    return (float(num) + k) / (float(denom) + (float(numVals) * k))

def find_images_in_class(data, labels, cls):
    """ Find images that belong to a specific class """
    rows = []
    count = 0
    for row in labels:
        if row == cls:
            rows.append(data[count])
        count += 1
    return rows

def parse_images(imageFile):
    """ Parse image files. Files _must_ represent images of 28x28 pixels """
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
    """ Parse the labels. These are assumed to be integers (one per line) """
    with open(labelFile, 'r') as f:
        return [int(x) for x in f]

def parse_args():
    parser = argparse.ArgumentParser(description="Classify digits")
    parser.add_argument("-t", "--training-data", required=True)
    parser.add_argument("-l", "--training-labels", required=True)
    parser.add_argument("-d", "--test-data", required=True)
    parser.add_argument("-f", "--test-labels", required=True)
    parser.add_argument("--overlap", dest="overlap", action="store_true")
    parser.add_argument("--no-overlap", dest="overlap", action="store_false")
    parser.add_argument("-r", "--group-rows", default=1, type=int)
    parser.add_argument("-c", "--group-cols", default=1, type=int)
    parser.set_defaults(overlap=True)
    return parser.parse_args()

if __name__ == "__main__":
    main()
