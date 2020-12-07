from load_write_data import outputResult
from evaluation import get_results
from bm25 import get_bm25_scores, get_bm25_mapback
import json
import pickle
import sys
import argparse
import os
import pandas as pd

# invoke this function as:
# python train.py ./inputFolder/ -k1 k1Val -b bVal -e epsilonVal
# OR invoke using deffault k1, b, epsilon arguments
# python train.py

# parse argparse argument for hyperparam tuning
parser = argparse.ArgumentParser()
parser.add_argument('inPath', nargs='?', type=str, help="input folder for raw dataset")
parser.add_argument('outPath', nargs='?', type=str, help="output folder for processed dataset")
parser.add_argument("-k1", nargs='?', default=1.2, type=float, help="hyperparam for bm25: k1")
parser.add_argument("-b", nargs='?', default=0.75, type=float, help="hyperparam for bm25: b")
parser.add_argument("-e", nargs='?', default=1e-8, type=float, help="hyperparam for bm25: epsilon")
parser.add_argument("--evaluate", default=False, action='store_true')
args = parser.parse_args()

inputPath = args.inPath
outputPath = args.outPath
k1 = args.k1
b = args.b
epsilon = args.e


# preprocess / load data data

def load_data(fn):
    print("load_data is file? " + str(os.path.isfile(fn)))
    df = pd.read_json(open(fn, "r", encoding="utf8"))
    data = df.to_dict(orient='list')
    return data

#def loadData(fn):
#    return pd.read_json(fn, orient='records').to_dict(orient='list')

train = load_data(os.path.join(inputPath, 'train-preprocessed.json'))
dev = load_data(os.path.join(inputPath, 'dev-preprocessed.json'))

if not os.path.exists(outputPath):
    os.mkdir(outputPath)

# get bm25 score
if not args.evaluate:
    print("Training bm25 with hyperparams: k1 = {}\t b = {}\t epsilon={}\t".format(k1, b, epsilon))
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
else:
    devScores, devLabels = get_bm25_scores(dev, k1, b, epsilon)

    # rank
    dev_results = get_results(devScores, devLabels, 'dev [bm25]')
    
    # writeout results
    outputResult(dev_results)
    
    # output to outfile
    resDev = get_bm25_mapback(dev, devScores, devLabels)
    with open(outputPath+'dev.json', 'w') as f:
        json.dump(resDev, f)

