import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.window = args.mlp_window
        self.linear1 = nn.Linear(args.kdim * args.mlp_window, args.mlp_hidden1, device=0)
        self.linear2 = nn.Linear(args.mlp_hidden1, args.mlp_hidden2, device=0)
        self.linear3 = nn.Linear(args.mlp_hidden2, 2, device=0)
        self.activate = nn.ReLU()
        self.dropout = nn.Dropout(p=args.dropout)
        self.dropoutRate = 1 - args.dropout

    def forward(self, x, stage):
        xStack = []
        while x.size(0) < self.window:
            x = torch.cat((x, x), dim=0)
            x = x[:self.window]
        for i in range(x.size(0) - self.window + 1):
            xStack.append(x[i:i + self.window].view(-1))
        x = torch.stack(xStack, dim=0)
        # print(x.size())
        x = self.linear1(x)
        x = self.activate(x)
        x = self.linear2(x)
        x = torch.mean(x, dim = 0).unsqueeze(0)
        x = self.activate(x)
        if stage == "train":
            x = self.dropout(x)
        else:
            x *= self.dropoutRate
        x = self.linear3(x)
        return x[0]