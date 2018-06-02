import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import *


class Bottleneck(nn.Module):
    def __init__(self, in_c, growthRate):
        super(Bottleneck, self).__init__()
        out_c = 4*growthRate
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, in_c, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out



class Conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3,
                              padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)


    def forward(self, x):
        out = self.conv(x)
        out = F.relu(self.bn(out))
        return out

class ConvNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(ConvNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2


        in_c = 2048

        self.dense1 = self._make_dense(2048, growthRate, nDenseBlocks, bottleneck)
        in_c += nDenseBlocks*growthRate
        out_c = int(math.floor(in_c*reduction))

        in_c = out_c
        out_c = int(in_c / 2)
        self.conv1 = Conv(in_c, out_c)


        in_c = out_c
        self.dense2 = self._make_dense(in_c, growthRate, nDenseBlocks, bottleneck)
        in_c += nDenseBlocks*growthRate
        out_c = int(math.floor(in_c*reduction))

        in_c = out_c
        out_c = int(in_c / 2)
        self.conv2 = Conv(in_c, out_c)


        in_c = out_c
        self.dense3 = self._make_dense(in_c, growthRate, nDenseBlocks, bottleneck)
        in_c += nDenseBlocks*growthRate


        self.bn = nn.BatchNorm2d(in_c)

        r = args.im_row / args.n_pool
        c = args.im_col / args.n_pool
        self.last_c = in_c * r * c
        self.fc = nn.Linear(self.last_c, args.n_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                '''n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))'''
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, in_c, growthRate, nDenseBlocks, bottleneck):

        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(in_c, growthRate))
            else:
                layers.append(SingleLayer(in_c, growthRate))
            in_c += growthRate
        return nn.Sequential(*layers)



    def forward(self, x):

        out = self.dense1(x)
        out = self.conv1(out)
        out = self.dense2(out)
        out = self.conv2(out)
        out = self.dense3(out)

        out = F.avg_pool2d(F.relu(self.bn(out)), 7)

        out = out.view(-1, self.last_c)
        out = F.log_softmax(self.fc(out))

        return out

if __name__ == '__main__':
    train_net = ConvNet(growthRate=24, depth=10, reduction=1,
                         bottleneck=True, nClasses=100)
    train_net.eval()
    a = np.ones((1, 2048, 7, 7))
    inputs = torch.Tensor(a)
    o = train_net(inputs)
    print(o)

