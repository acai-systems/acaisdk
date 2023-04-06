""" 
Simple example to run MNIST CNN on cluster
"""
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import sys
output_dir = sys.argv[1]
epoch_num = sys.argv[2]
fp = open(output_dir + "./loss.txt", "w")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization
    
def train(num_epochs, cnn, train_loader):
    
    # Train the model
    cnn.train()
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Train based on batches
            b_x = Variable(images) 
            b_y = Variable(labels)  
            output = cnn(b_x)[0]               
            loss = loss_func(output, b_y)
            optimizer.zero_grad()           
            loss.backward()    
            optimizer.step()        

            if (i+1) % 100 == 0:                
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}\n' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                fp.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}\n' 
                       .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            pass


train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

from torch.utils.data import DataLoader
train_loader = torch.utils.data.DataLoader(train_data, 
                                          batch_size=100, 
                                          shuffle=True),
    

cnn = CNN()
loss_func = nn.CrossEntropyLoss()   
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)   

num_epochs = 10
