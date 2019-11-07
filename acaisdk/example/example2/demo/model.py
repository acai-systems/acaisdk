import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):

    def __init__(self, channel_size, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_size)
        self.shortcut = nn.Conv2d(channel_size, channel_size, kernel_size=1, stride=stride, bias=False)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Network(nn.Module):
    def __init__(self, num_feats, hidden_sizes, num_classes, feat_dim=10):
        super(Network, self).__init__()
        
        self.hidden_sizes = [num_feats] + hidden_sizes + [num_classes]
        
        self.layers = []
        for idx, channel_size in enumerate(hidden_sizes):
            self.layers.append(nn.Conv2d(in_channels=self.hidden_sizes[idx], 
                                        out_channels=self.hidden_sizes[idx+1], 
                                        kernel_size=3, stride=2, bias=False))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(BasicBlock(channel_size = channel_size))
            
        self.layers = nn.Sequential(*self.layers)
        self.linear_label = nn.Linear(self.hidden_sizes[-2], self.hidden_sizes[-1], bias=False)
        
        # For creating the embedding to be passed into the Center Loss criterion
        self.linear_closs = nn.Linear(self.hidden_sizes[-2], feat_dim, bias=False)
        self.relu_closs = nn.ReLU(inplace=True)
    
    def forward(self, x, evalMode=False):
        output = x
        output = self.layers(output)
            
        output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
        output = output.reshape(output.shape[0], output.shape[1])
        
        label_output = self.linear_label(output)
        label_output = label_output/torch.norm(self.linear_label.weight, dim=1)
        
        # Create the feature embedding for the Center Loss
        closs_output = self.linear_closs(output)
        closs_output = self.relu_closs(closs_output)

        return closs_output, label_output