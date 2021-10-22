import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
# import torch.backends.cudnn as cudnn

import ray
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.utils import AverageMeterCollection, BATCH_SIZE

import os
import time
import argparse
import sys

# from models import *

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        # print("\tIn Model: input size", x.size(),
        #       "output size", out.size(), torch.cuda.is_available(), "num of GPUs:", torch.cuda.device_count(), "current:", torch.cuda.current_device())

        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])


def model_creator(config):
    """Returns a torch.nn.Module object."""
    net = ResNet18()
    # if config["use_gpu"]:
    #     net = torch.nn.DataParallel(net)
        # cudnn.benchmark = True
    return net


def optimizer_creator(model, config):
    """Returns optimizer defined upon the model parameters."""
    return torch.optim.SGD(model.parameters(), lr=config["lr"],
                      momentum=0.9, weight_decay=5e-4)


def scheduler_creator(optimizer, config):
    """Returns a learning rate scheduler wrapping the optimizer.
    """
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def data_creator(config):
    """Returns training dataloader, validation dataloader."""
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
    if config["num_batch"] != -1:
        subset = list(range(0, config["batch_size"] * config["num_batch"]))
        trainset = torch.utils.data.Subset(trainset, subset)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config["batch_size"], shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader


def loss_creator(config):
    return nn.CrossEntropyLoss()


class MyTrainingOperator(TrainingOperator):
    def setup(self, config):
        # Setup all components needed for training here. This could include
        # data, models, optimizers, loss & schedulers.

        # Setup data loaders.
        train_loader, val_loader = data_creator(config)

        # Setup model.
        model = model_creator(config)

        # Setup optimizer.
        optimizer = optimizer_creator(model, config)

        # Setup loss.
        criterion = loss_creator(config)

        # Setup scheduler.
        scheduler = scheduler_creator(optimizer, config)

        # Register all of these components with Ray SGD.
        # This allows Ray SGD to do framework level setup like Cuda, DDP,
        # Distributed Sampling, FP16.
        # We also assign the return values of self.register to instance
        # attributes so we can access it in our custom training/validation
        # methods.
        self.model, self.optimizer, self.criterion, self.scheduler = \
            self.register(models=model, optimizers=optimizer,
                          criterion=criterion,
                          schedulers=scheduler)
        # self.new_model = nn.DataParallel(model)
        self.register_data(train_loader=train_loader, validation_loader=val_loader)

    # def train_epoch(self, iterator, info):
    #     # use_gpu = True if info["device"] == "cuda" else False
    #     # print("GPU:", use_gpu, "num of GPUs:", torch.cuda.device_count(), 
    #     #   "current:", torch.cuda.current_device(), "name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    #     # print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    #     # print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))

    #     meter_collection = AverageMeterCollection()
    #     self.model.train()
    #     train_loss = 0
    #     correct = 0
    #     total = 0
    #     for batch_idx, (inputs, targets) in enumerate(iterator):
            
    #         # inputs, targets = inputs.to(info["device"]), targets.to(info["device"])
    #         self.optimizer.zero_grad()
    #         outputs = self.model(inputs)
    #         # outputs = self.new_model(inputs)
    #         loss = self.criterion(outputs, targets)
    #         loss.backward()
    #         self.optimizer.step()

    #         train_loss += loss.item()
    #         _, predicted = outputs.max(1)
    #         total += targets.size(0)
    #         correct += predicted.eq(targets).sum().item()

    #         # print("Batch:", batch_idx, "of", info["num_batch"], 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
    #         #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    #         # sys.stdout.flush()

    #         metrics = {"batch_idx": batch_idx, "loss": train_loss/(batch_idx+1), "acc": 100.*correct/total}


    #         # print("--- batch", batch_idx, "---")
    #         print("batch:", batch_idx, "Outside: input size", inputs.size(),
    #             "output_size", outputs.size())

    #     self.scheduler.step()

    #     # Returns stats of the meters.
    #     stats = meter_collection.summary()
    #     return stats


def train_example(lr, use_gpu, num_epoch, batch_size, num_batch, num_worker):
    epoch_times = []
    
    # init trainer
    trainer1 = TorchTrainer(
        training_operator_cls=MyTrainingOperator,
        num_workers=num_worker,
        use_gpu=use_gpu,
        wrap_ddp=False,
        config={
            "lr": lr,  # used in optimizer_creator
            "batch_size": batch_size,
            "num_batch": num_batch,  # used in data_creator
            "use_gpu": use_gpu
            
        },
        # backend="gloo",
        scheduler_step_freq="epoch")
    
    info = {}
    info["num_batch"] = num_batch
    if use_gpu:
        info["device"] = 'cuda'
    else:
        info["device"] = 'cpu'

    # start train
    for i in range(num_epoch):
        print('\n------ Epoch: %d ------' % i)
        sys.stdout.flush()
        epoch_start = time.time()
        stats = trainer1.train(info=info)
        epoch_end = time.time()
        print(stats)
        print(f"Epoch time: {epoch_end - epoch_start} s")
        sys.stdout.flush()
        epoch_times.append(epoch_end - epoch_start)
    

    # validater
    # print(trainer1.validate())
    print(f"Average Epoch time: {sum(epoch_times) / len(epoch_times)} s")
    trainer1.shutdown()
    print("Training success! Shut down trainer.")
    sys.stdout.flush()



if __name__ == "__main__":
    program_start = time.time()
    # ini ray cluster
    if ("RAY_HEAD_SERVICE_HOST" not in os.environ
            or os.environ["RAY_HEAD_SERVICE_HOST"] == ""):
        raise ValueError("RAY_HEAD_SERVICE_HOST environment variable empty."
                         "Is there a ray cluster running?")
    # if ("NNODE" not in os.environ):
    #     raise ValueError("NNODE not in environment variable.")
    ray_head_service_host = os.environ["RAY_HEAD_SERVICE_HOST"]
    # arg_num_worker = int(os.environ["NNODE"]) + 1
    ray.init(address=ray_head_service_host + ":6379")
    # arg_num_worker = 0
    # ray.init()


    # parse arguments
    parser = argparse.ArgumentParser(description='Ray PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--gpu', default=1, type=int, help='use gpu or not')
    parser.add_argument('--num_epoch', default=1, type=int, help='number of epochs to train')
    parser.add_argument('--batch_size', default=128, type=int, help='number of epochs to train')
    parser.add_argument('--num_batch', default=20, type=int, help='number of batahs to use in training')
    # parser.add_argument('--num_worker', default=1, type=int, help='number of worker nodes in ray cluster')
    args = parser.parse_args()

    # lr
    arg_lr = args.lr
    # whether to use gpu
    arg_use_gpu = True if args.gpu > 0 and torch.cuda.is_available() else False
    device = "GPU" if arg_use_gpu else "CPU"
    arg_num_worker = int(ray.cluster_resources().get(device))
    # epoch
    arg_num_epoch = args.num_epoch
    # batch
    arg_batch_size = args.batch_size
    # num of batch
    arg_num_batch = args.num_batch
    # print configuration
    print("lr:", arg_lr)
    print("GPU:", arg_use_gpu, "num of GPUs:", torch.cuda.device_count(), 
          "current:", torch.cuda.current_device(), "name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    
    print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
    # print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    
    print("num_epoch:", arg_num_epoch)
    print("batch_size:", arg_batch_size)
    print("num_batch:", arg_num_batch)
    print("num_worker:", arg_num_worker)
    sys.stdout.flush()


    # train model
    train_start = time.time()
    train_example(arg_lr, arg_use_gpu, arg_num_epoch, arg_batch_size, arg_num_batch, arg_num_worker)
    train_end = time.time()

    print(f"Training time: {train_end - train_start} s")
    print(f"Program time: {train_end - program_start} s")
    print("###### End of program ######")
    sys.stdout.flush()