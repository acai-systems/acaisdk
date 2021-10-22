import argparse
import os
import numpy as np
from PIL import Image
from model import Network, BasicBlock

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

print("-- Entering train.py")

parser = argparse.ArgumentParser(
    description='A simple CNN for image classfication')
parser.add_argument(
    "--train_data",
    help="location of training data",
    type=str)
parser.add_argument(
    "--dev_data",
    help="location of validation data",
    type=str)
parser.add_argument("--output_folder", help="location of output data")
parser.add_argument("--hidden", nargs="+", help="hidden layers", type=int)
parser.add_argument("--epoch", help="# of epochs to run", type=int)
parser.add_argument("--lr", help="learnin rate", type=int)
parser.add_argument("--batch", help="batch size", type=int)

# Global var used in train and test
# device = None


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)


def train(
        model,
        data_loader,
        test_loader,
        numEpochs,
        task='Classification'):
    print("-- ready to rock to model.train()")
    model.train()
    print("-- finish model.train()")

    print("-- ready to rock to training epochs")

    print("numer of epochs:", numEpochs)
    for epoch in range(numEpochs):
        print("epoch:", epoch)
        avg_loss = 0.0
        print("size:", len((data_loader)))
        for batch_num, (feats, labels) in enumerate(data_loader):
            print(batch_num, feats, labels)
            feats, labels = feats.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(feats)[1]

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            print("here 1")

            # print('train Epoch: {}\tBatch: {}\tLoss: {:.4f}'.format(
            #     epoch + 1, batch_num + 1, loss.item()))

            if batch_num % 50 == 49:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch + 1,
                                                                      batch_num + 1, avg_loss / 50))
                avg_loss = 0.0

            print("here 1")

            torch.cuda.empty_cache()
            del feats
            del labels
            del loss

            print("here 3")

        # TODO: there's some bug here that lead the program to exit with error
        # if task == 'Classification':
        #     val_loss, val_acc = test_classify(model, test_loader)
        #     train_loss, train_acc = test_classify(model, data_loader)
        #     print(
        #         'Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'. format(
        #             train_loss,
        #             train_acc,
        #             val_loss,
        #             val_acc))


def test_classify(model, test_loader):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0

    print("-- ready to rock to testing batches")

    for batch_num, (feats, labels) in enumerate(test_loader):
        print("hit 0")
        feats, labels = feats.to(device), labels.to(device)
        print("hit A")
        outputs = model(feats)[1]
        print("hit B")
        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
        print("hit C")
        pred_labels = pred_labels.view(-1)
        print("hit D")

        loss = criterion(outputs, labels.long())

        print('test Batch: {}\tLoss: {:.4f}'.format(
            batch_num + 1, loss.item()))

        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
        test_loss.extend([loss.item()] * feats.size()[0])
        del feats
        del labels

    model.train()
    return np.mean(test_loss), accuracy / total


if __name__ == '__main__':
    train_foler = "./"
    dev_foler = "./"
    output_folder = "./output"
    lr = 1e-2
    weightDecay = 5e-5
    numEpochs = 1
    num_feats = 3
    hidden_sizes = [5, 10]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    if args.train_data:
        train_foler = args.train_data
        print("-- set training folder to " + train_foler)
    if args.dev_data:
        dev_foler = args.dev_data
        print("-- set validation folder to " + dev_foler)
    if args.hidden:
        hidden_sizes = args.hidden
    if args.epoch:
        numEpochs = args.epoch
        print("-- num epochs:", numEpochs)
    if args.lr:
        lr = args.lr
    if args.output_folder:
        output_folder = args.output_folder

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("-- making output folder: " + output_folder)

    train_dataset = torchvision.datasets.ImageFolder(
        root=train_foler, transform=torchvision.transforms.ToTensor())
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=10, shuffle=True, num_workers=1)

    dev_dataset = torchvision.datasets.ImageFolder(
        root=dev_foler, transform=torchvision.transforms.ToTensor())
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=10, shuffle=True, num_workers=1)
    num_classes = len(train_dataset.classes)
    network = Network(num_feats, hidden_sizes, num_classes)
    network.apply(init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        network.parameters(),
        lr=lr,
        weight_decay=weightDecay,
        momentum=0.9)
    print("-- ready to rock to network.train")
    network.train()
    print("-- finish network.train")
    network.to(device)
    print("-- ready to rock to train()")
    train(network, train_dataloader, dev_dataloader, numEpochs)
    print("-- finish train() ***************** ")
    torch.save(network.cpu(), output_folder + "/network.npy")
