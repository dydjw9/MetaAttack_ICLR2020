import torch.nn as nn
import torch.nn.functional as F

class zoo_attack(nn.Module):
    def __init__(self):
        super(zoo_attack, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.fc1 = nn.Linear(5*5*128, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        #print(x.shape)
        x = x.view(-1, 5*5*128)
        x = F.dropout(F.relu(self.fc1(x)), 0.5)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x