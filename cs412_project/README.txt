The code is a Python 3 file. It can be executed by running
'python project/project_main.py trainingfile testingfile [outputfile]'
from within the main folder (not the project or results subfolders).

If no output file is specified, the default is results/submission.csv

So, for this project, the program should be run using
'python project/project_main.py results/training.csv results/testing.csv'
from within the main folder.

Predictions are written to the output file specified on the command line,
or to results/submission.csv if none is specified.

The code uses the modules sys, math, operator, numpy, and random.
