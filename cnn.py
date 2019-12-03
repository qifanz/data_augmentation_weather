import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torch.optim as optim
from torchvision import models
import torch.cuda

use_cuda=True

torch.manual_seed(2) #500 #400 #100
torch.cuda.manual_seed(8) #700 #600 #150

TRAIN_DATA_PATH = "trainsnow"
TEST_DATA_PATH = "testsnow"

dir_path = os.path.dirname(os.path.realpath(__file__))
BATCH_SIZE = 32
transform = transforms.Compose([transforms.Resize(size=(64,64)),transforms.ToTensor()])
train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH,transform=transform)
train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)


class Net(nn.Module):
    def __init__(self, class_num = 751):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 10)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)#, momentum=0.9)

for epoch in range(300):  # loop over the dataset multiple times
    running_loss = 0.0
    total=0
    correct=0
    for i, datum in enumerate(train_data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = datum
        inputs=inputs.to('cuda')
        labels=labels.to('cuda')
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        #print(torch.topk(outputs.data,3,dim=1)[1])
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # print statistics
        running_loss += loss.item()
    print('accuracy : ',100*correct/total)
    print(running_loss)

    
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH,transform=transform)
test_data_loader  = data.DataLoader(test_data, shuffle=True)

total=0
correct=0
for i, datum in enumerate(test_data_loader,0):
    inp, lab = datum
    inp=inp.to('cuda')
    lab=lab.to('cuda')
    outputs = net(inp)
    _, predicted = torch.max(outputs.data, 1)
    total += lab.size(0)
    correct += (predicted == lab).sum().item()
print('accuracy test : ',100*correct/total)
