import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.featureMaps3 = nn.Conv2d(1, 100, (3, args.kdim), device=0)
        self.featureMaps4 = nn.Conv2d(1, 100, (4, args.kdim), device=0)
        self.featureMaps5 = nn.Conv2d(1, 100, (5, args.kdim), device=0)
        self.dropout = nn.Dropout(p=args.dropout)
        self.dropoutRate = args.dropout
        self.fullyConnectedLayer = nn.Linear(300, 2, device=0)

    def forward(self, x, stage):
        x = x.to(device=0)
        print(x)
        x = torch.unsqueeze(x, 0)
        x1 = self.featureMaps3(x)
        x2 = self.featureMaps4(x)
        x3 = self.featureMaps5(x)
        print(x1.size())
        print(x2.size())
        print(x3.size())
        x1, _indices1 = torch.max(x1, 1)
        x2, _indices2 = torch.max(x2, 1)
        x3, _indices3 = torch.max(x3, 1)
        x = torch.cat((x1, x2, x3), 0)
        x = torch.swapaxes(x, 0, 1)
        if stage == "train":
            x = self.dropout(x)
        else:
            x *= self.dropoutRate
        x = self.fullyConnectedLayer(x)
        print(x.size())