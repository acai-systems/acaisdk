from preprocess_squad import preprocess
from load_write_data import outputResult
from evaluation import get_results
from bm25 import get_bm25_scores, get_bm25_mapback
import json
import sys

inputPath = sys.argv[1]
outputPath = sys.argv[2]


# preprocess / load data data
train, dev = preprocess(path=inputPath, newPrepocess=True)

# get bm25 score
trainScores, trainLabels = get_bm25_scores(train)
devScores, devLabels = get_bm25_scores(dev)

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