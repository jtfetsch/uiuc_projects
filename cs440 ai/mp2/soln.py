import pickle
from mp2 import State

data = None
with open('state.pickle', 'r') as f:
    data = pickle.load(f)

words = []
with open('banks/bank3.txt', 'r') as f:
    words = [x.upper().strip() for x in f]

for x in data:
    if type(x) == list:
        for (_, assignment) in x:
            state = State([['_' for x in range(9)] for y in range(9)], set())
            print "Words: {}".format(len(assignment))
            for a in assignment:
                (word, val) = a
                (dir, (row, col)) = val
                state.makeAssignment(word, val)
                print "{}, {}, {}: {}".format(dir, row, col, word)
            aWords = map(lambda x: x[0], assignment)
            for word in words:
                if word not in aWords:
                    print "NA: {}".format(word)
            state.printBoard()
    if type(x) == int:
            print "Expanded: {}".format(x)
            print "--------"
