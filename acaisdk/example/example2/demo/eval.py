import argparse
import os
import csv
import numpy as np
from PIL import Image
from model import Network, BasicBlock

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description='evaluating a dataset')
parser.add_argument("--eval_data", help="location of eval data", type=str)
parser.add_argument("--model", help="location of model", type=str)

def test_classify(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    for batch_num, (feats, labels) in enumerate(test_loader):
        feats, labels = feats.to(device), labels.to(device)
        outputs = model(feats)[1]
            
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        pred_labels = pred_labels.view(-1)
            
        loss = criterion(outputs, labels.long())
            
        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()]*feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy/total

if __name__ == '__main__':
    eval_data = "./"
    model = "./"
    output_folder = "./eval_output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    args = parser.parse_args()
    if args.eval_data:
        eval_data = args.eval_data
    if args.model:
        model = args.model
    network = torch.load(model)
    test_dataset = torchvision.datasets.ImageFolder(root=eval_data,
                                                transform=torchvision.transforms.ToTensor())
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, 
                                                shuffle=False, num_workers=8)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    eval_loss, eval_acc = test_classify(network, test_dataloader)
    print('Eval Loss: {:.4f}\tEval Accuracy: {:.4f}'.format(eval_loss, eval_acc))
    with open("./eval_output/eval_result.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow([eval_loss, eval_acc])