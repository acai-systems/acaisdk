from preprocess_squad import preprocess
import json
import sys
import argparse

# invoke this function as:
# python preprocess.py ./inputFolder/


# parse argparse argument for hyperparam tuning
parser = argparse.ArgumentParser()
parser.add_argument('inPath', nargs='?', type=str, help="input folder for raw dataset")
parser.add_argument('outPath', nargs='?', type=str, help="output folder for raw dataset")
args = parser.parse_args()

inputPath = args.inPath
outputPath = args.outPath


# preprocess / load data data
train, dev = preprocess(inputPath, outputPath, newPrepocess=True)
