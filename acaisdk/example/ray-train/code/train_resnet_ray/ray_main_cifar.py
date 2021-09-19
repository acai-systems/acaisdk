import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from filelock import FileLock
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from tqdm import trange

import ray
from ray.util.sgd.torch import TorchTrainer, TrainingOperator
# from ray.util.sgd.torch.resnet import ResNet18
from ray.util.sgd.utils import BATCH_SIZE, override

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))

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
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.BatchNorm2d(self.expansion * planes))

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

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
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
        print("\tIn Model: input size", x.size(),
              "output size", out.size(), torch.cuda.is_available(), "num of GPUs:", torch.cuda.device_count(), "current:", torch.cuda.current_device())

        return out


def ResNet18(_):
    return ResNet(BasicBlock, [2, 2, 2, 2])


def initialization_hook():
    # Need this for avoiding a connection restart issue on AWS.
    os.environ["NCCL_SOCKET_IFNAME"] = "^docker0,lo"
    os.environ["NCCL_LL_THRESHOLD"] = "0"

    # set the below if needed
    # print("NCCL DEBUG SET")
    # os.environ["NCCL_DEBUG"] = "INFO"


class CifarTrainingOperator(TrainingOperator):
    @override(TrainingOperator)
    def setup(self, config):
        # Create model.
        model = ResNet18(config)

        # Create optimizer.
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.get("lr", 0.1),
            momentum=config.get("momentum", 0.9))

        # Load in training and validation data.
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])  # meanstd transformation

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        with FileLock(".ray.lock"):
            train_dataset = CIFAR10(
                root="~/data",
                train=True,
                download=True,
                transform=transform_train)
            validation_dataset = CIFAR10(
                root="~/data",
                train=False,
                download=False,
                transform=transform_test)

        if config["test_mode"]:
            train_dataset = Subset(train_dataset, list(range(64)))
            validation_dataset = Subset(validation_dataset, list(range(64)))

        train_loader = DataLoader(
            train_dataset, batch_size=config[BATCH_SIZE], num_workers=2)
        validation_loader = DataLoader(
            validation_dataset, batch_size=config[BATCH_SIZE], num_workers=2)

        # Create scheduler.
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[150, 250, 350], gamma=0.1)

        # Create loss.
        criterion = nn.CrossEntropyLoss()

        # Register all components.
        self.model, self.optimizer, self.criterion, self.scheduler = \
            self.register(models=model, optimizers=optimizer,
                          criterion=criterion, schedulers=scheduler)
        self.register_data(
            train_loader=train_loader, validation_loader=validation_loader)


def train_cifar(test_mode=False,
                num_workers=1,
                use_gpu=False,
                num_epochs=5,
                fp16=False):
    trainer1 = TorchTrainer(
        training_operator_cls=CifarTrainingOperator,
        initialization_hook=initialization_hook,
        num_workers=num_workers,
        config={
            "lr": 0.1,
            "test_mode": test_mode,  # subset the data
            # this will be split across workers.
            BATCH_SIZE: 128 * num_workers
        },
        use_gpu=use_gpu,
        scheduler_step_freq="epoch",
        use_fp16=fp16,
        use_tqdm=False)
    pbar = trange(num_epochs, unit="epoch")
    for i in pbar:
        info = {"num_steps": 1} if test_mode else {}
        info["epoch_idx"] = i
        info["num_epochs"] = num_epochs
        # Increase `max_retries` to turn on fault tolerance.
        trainer1.train(max_retries=1, info=info)
        # val_stats = trainer1.validate()
        # pbar.set_postfix(dict(acc=val_stats["val_accuracy"]))

    # print(trainer1.validate())
    trainer1.shutdown()
    print("success!")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--address",
    #     required=False,
    #     type=str,
    #     help="the address to use for connecting to the Ray cluster")
    # parser.add_argument(
    #     "--server-address",
    #     type=str,
    #     default=None,
    #     required=False,
    #     help="The address of server to connect to if using "
    #     "Ray Client.")
    # parser.add_argument(
    #     "--num-workers",
    #     "-n",
    #     type=int,
    #     default=1,
    #     help="Sets number of workers for training.")
    # parser.add_argument(
    #     "--num-epochs", type=int, default=5, help="Number of epochs to train.")
    # parser.add_argument(
    #     "--use-gpu",
    #     action="store_true",
    #     default=False,
    #     help="Enables GPU training")
    # parser.add_argument(
    #     "--fp16",
    #     action="store_true",
    #     default=False,
    #     help="Enables FP16 training. Requires `use-gpu`.")
    # parser.add_argument(
    #     "--smoke-test",
    #     action="store_true",
    #     default=False,
    #     help="Finish quickly for testing.")
    # parser.add_argument(
    #     "--tune", action="store_true", default=False, help="Tune training")

    # args, _ = parser.parse_known_args()

    # if args.server_address:
    #     ray.init(f"ray://{args.server_address}")
    # else:
    #     num_cpus = 4 if args.smoke_test else None
    #     ray.init(address=args.address, num_cpus=num_cpus, log_to_driver=True)
    if ("RAY_HEAD_SERVICE_HOST" not in os.environ
            or os.environ["RAY_HEAD_SERVICE_HOST"] == ""):
        raise ValueError("RAY_HEAD_SERVICE_HOST environment variable empty."
                         "Is there a ray cluster running?")
    redis_host = os.environ["RAY_HEAD_SERVICE_HOST"]
    ray.init(address=redis_host + ":6379")

    train_cifar(
        # test_mode=args.smoke_test,
        # num_workers=args.num_workers,
        num_workers=2,
        use_gpu=True,
        # num_epochs=args.num_epochs,
        num_epochs=1,
        fp16=False)