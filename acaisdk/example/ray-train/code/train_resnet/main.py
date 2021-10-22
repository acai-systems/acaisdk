'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import time

import sys

from models import *
# from utils import progress_bar

program_start = time.time()


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--gpu', default=1, type=int, help='use gpu or not')
parser.add_argument('--num_epoch', default=1, type=int, help='number of epochs to train')
parser.add_argument('--batch_size', default=128, type=int, help='number of epochs to train')
parser.add_argument('--num_batch', default=20, type=int, help='number of batahs to use in training')
args = parser.parse_args()


print(args.gpu)
# whether to use gpu
device = 'cpu'
if args.gpu > 0 and torch.cuda.is_available():
    device = 'cuda'
# epoch
num_epoch = args.num_epoch
# batch
batch_size = args.batch_size
# num of batch
num_batch = args.num_batch
# print configuration
print("Device:", device)
print("num of GPUs:", torch.cuda.device_count(), 
      "current:", torch.cuda.current_device(), "name:", torch.cuda.get_device_name(torch.cuda.current_device()))
print("num_epoch:", num_epoch)
print("batch_size:", batch_size)
print("num_batch:", num_batch)
sys.stdout.flush()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
end_epoch = start_epoch + num_epoch

# Data
print('==> Preparing data..')
sys.stdout.flush()
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
if num_batch != -1:
    subset = list(range(0, batch_size * args.num_batch))
    trainset = torch.utils.data.Subset(trainset, subset)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
sys.stdout.flush()
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    sys.stdout.flush()
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    sys.stdout.flush()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # print("batch:", batch_idx, inputs.size(), targets.size())
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # print("Batch:", batch_idx, "of", len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # sys.stdout.flush()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


# def test(epoch):
#     global best_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)

#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#             print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                          % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

#             # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#             #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

#     # Save checkpoint.
#     acc = 100.*correct/total
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir('checkpoint'):
#             os.mkdir('checkpoint')
#         torch.save(state, './checkpoint/ckpt.pth')
#         best_acc = acc

epoch_times = []
train_start = time.time()
for epoch in range(start_epoch, end_epoch):
    epoch_start = time.time()
    train(epoch)
    epoch_end = time.time()
    epoch_times.append(epoch_end - epoch_start)
    print(f"Runtime of the Epoch is {epoch_end - epoch_start} s")
    sys.stdout.flush()
    # test(epoch)
    scheduler.step()
train_end = time.time()
print("finish train")
# avg_time = sum(epoch_times) / len(epoch_times)
print(f"Average runtime of each epoch is {sum(epoch_times) / len(epoch_times)} s")
print(f"Training time: {train_end - train_start} s")
print(f"Program time: {train_end - program_start} s")
print("###### End of program ######")
sys.stdout.flush()

# import os
# print("files:")
# files = [f for f in os.listdir('.')]
# for f in files:
#     print(f)
# print("-------")

# f = open("/ray_output/train_resnet_output.txt", "w") 
# f.write("job finish")
# f.close()

# print("finish write output")


