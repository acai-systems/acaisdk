'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator

import os
import ray

from ray.util.sgd.utils import AverageMeterCollection

import time

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
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

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
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
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])


def model_creator(config):
    """Returns a torch.nn.Module object."""
    return ResNet18()


def optimizer_creator(model, config):
    """Returns optimizer defined upon the model parameters."""
    return torch.optim.SGD(model.parameters(),
                      lr=config.get("lr", 1e-2), momentum=0.9, weight_decay=5e-4)


def scheduler_creator(optimizer, config):
    """Returns a learning rate scheduler wrapping the optimizer.
    You will need to set ``TorchTrainer(scheduler_step_freq="epoch")``
    for the scheduler to be incremented correctly.
    If using a scheduler for validation loss, be sure to call
    ``trainer.update_scheduler(validation_loss)``.
    """
    epochs = config.get("epochs", 1)
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.0133 ** (1.0 / epochs))
    # return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)


def data_creator(config):
    """Returns training dataloader, validation dataloader."""
    # train_dataset = LinearDataset(2, 5, size=config.get("data_size", 1000))
    # val_dataset = LinearDataset(2, 5, size=config.get("val_size", 400))
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=config.get("batch_size", 32),
    # )
    # validation_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=config.get("batch_size", 32))
    # return train_loader, validation_loader
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

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(dir_path, "data")

    batch_size =  config.get("batch_size", 128)

    trainset = torchvision.datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=transform_train)
    subset = list(range(0, 128 * 20))
    trainset_1 = torch.utils.data.Subset(trainset, subset)
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.get("batch_size", 32), shuffle=True)
    train_loader = torch.utils.data.DataLoader(trainset_1, batch_size=batch_size, shuffle=False)
    # train_loader.autoscale_batch_size(4096, local_bsz_bounds=(32, 1024),
    #                              gradient_accumulation=True)

    validset = torchvision.datasets.CIFAR10(root=dataset_path, train=False, download=False, transform=transform_test)
    # validation_loader = torch.utils.data.DataLoader(validset, batch_size=config.get("batch_size", 32))
    validation_loader = torch.utils.data.DataLoader(validset, batch_size=100)

    return train_loader, validation_loader


def loss_creator(config):
    return nn.CrossEntropyLoss()


class ResnetTrainingOperator(TrainingOperator):
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
        self.register_data(train_loader=train_loader, validation_loader=val_loader)

    def train_epoch(self, iterator, info):
        # print('\nEpoch: %d' % epoch)
        train_loss = 0
        correct = 0
        total = 0
        meter_collection = AverageMeterCollection()
        self.model.train()

        # print("total_batch:", len(list(iterator)))
        
        for batch in enumerate(iterator):
            batch_idx, (inputs, targets) = batch
            print("batch:", batch_idx)
            inputs, targets = inputs.to('cpu'), targets.to('cpu')
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # met_str = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total)

            met_loss = train_loss/(batch_idx+1)
            met_acc = 100.*correct/total

            print(met_loss, met_acc)

            # do some processing
            metrics = {"batch_idx": batch_idx, "loss": met_loss, "acc": met_acc} # dict of metrics

            # This keeps track of all metrics across multiple batches
            meter_collection.update(metrics, n=len(batch))
        
        self.scheduler.step()

        # Returns stats of the meters.
        stats = meter_collection.summary()
        return stats


def train_example(num_workers=1, use_gpu=False):
    trainer1 = TorchTrainer(
        training_operator_cls=ResnetTrainingOperator,
        num_workers=num_workers,
        use_gpu=use_gpu,
        config={
            "lr": 0.08,  # used in optimizer_creator
            "batch_size": 128,  # used in data_creator
            "epochs": 50
        },
        backend="gloo",
        scheduler_step_freq="epoch")


    for i in range(1):
        stats = trainer1.train()
        print(stats)

    # print(trainer1.validate())
    trainer1.shutdown()
    print("success!")


if __name__ == "__main__":
    if ("RAY_HEAD_SERVICE_HOST" not in os.environ
            or os.environ["RAY_HEAD_SERVICE_HOST"] == ""):
        raise ValueError("RAY_HEAD_SERVICE_HOST environment variable empty."
                         "Is there a ray cluster running?")
    if ("NNODE" not in os.environ):
        raise ValueError("NNODE not in environment variable.")
    ray_head_service_host = os.environ["RAY_HEAD_SERVICE_HOST"]
    num_workers = int(os.environ["NNODE"])
    ray.init(address=ray_head_service_host + ":6379")
    start = time.time()
    train_example(num_workers=num_workers, use_gpu=False)
    end = time.time()
    print(f"Runtime of the program is {end - start} s")
