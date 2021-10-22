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

if __name__ == "__main__":
    if ("RAY_HEAD_SERVICE_HOST" not in os.environ
                or os.environ["RAY_HEAD_SERVICE_HOST"] == ""):
            raise ValueError("RAY_HEAD_SERVICE_HOST environment variable empty."
                            "Is there a ray cluster running?")
    if ("NNODE" not in os.environ):
        raise ValueError("NNODE not in environment variable.")
    ray_head_service_host = os.environ["RAY_HEAD_SERVICE_HOST"]
    arg_num_worker = int(os.environ["NNODE"]) + 1
    ray.init(address=ray_head_service_host + ":6379")


    device = "GPU"
    num_workers = int(ray.cluster_resources().get(device))
    print(num_workers)