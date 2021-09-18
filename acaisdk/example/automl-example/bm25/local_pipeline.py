from preprocess_squad import preprocess
from load_write_data import outputResult
from evaluation import get_results
from bm25 import get_bm25_scores, get_bm25_mapback
import json
import sys
import argparse

# parse argparse argument for hyperparam tuning
parser = argparse.ArgumentParser()
parser.add_argument('inPath', nargs='?', type=str, help="input folder for raw dataset")
parser.add_argument('outPath', nargs='?', type=str, help="output folder for processed dataset")
parser.add_argument("-k1", nargs='?', default=1.2, type=float, help="hyperparam for bm25: k1")
parser.add_argument("-b", nargs='?', default=0.75, type=float, help="hyperparam for bm25: b")
parser.add_argument("-e", nargs='?', default=1e-8, type=float, help="hyperparam for bm25: epsilon")
args = parser.parse_args()

inputPath = args.inPath
outputPath = args.outPath
k1 = args.k1
b = args.b
epsilon = args.e


# preprocess / load data data
train, dev = preprocess(path=inputPath, newPrepocess=True)

# get bm25 score
trainScores, trainLabels = get_bm25_scores(train, k1, b, epsilon)
devScores, devLabels = get_bm25_scores(dev, k1, b, epsilon)

# rank
train_results = get_results(trainScores, trainLabels, 'train [bm25]')
dev_results = get_results(devScores, devLabels, 'dev [bm25]')

# writeout results
outputResult(train_results)
outputResult(dev_results)

# output to outfile
resTrain = get_bm25_mapback(train, trainScores, trainLabels)
resDev = get_bm25_mapback(dev, devScores, devLabels)
with open(outputPath+'train.json', 'w') as f:
    json.dump(resTrain, f)
with open(outputPath+'dev.json', 'w') as f:
    json.dump(resDev, f)