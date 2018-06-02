import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pretrain_model.resnext_101_32x4d import *

from  datasets.deal import *
from torch.autograd import Variable
from densenet_fixed import DenseNet
from conv_net import ConvNet
from config import *


#train_loader, test_loader = gen_loader()
train_loader = gen_all_loader()

#训练模型
#train_net = DenseNet(growthRate=12, depth=22, reduction=0.5,
#                     bottleneck=True, nClasses=100)

'''train_net = ConvNet(growthRate=36, depth=121, reduction=1,
                     bottleneck=True, nClasses=100)'''


train_net = resnext_101_32x4d
train_net.load_state_dict(torch.load('pretrain_model/resnext_101_32x4d.pth'))
train_net = nn.Sequential(*list(train_net.children())[:-1])
to_add = nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),
                       nn.Linear(30720,100),
                       Lambda(lambda x: F.log_softmax(x)))


train_net.add_module('fc',to_add)





'''for i, param in enumerate(train_net.parameters()):
    if i>30 and i<280:
        param.weight.requires_grad = False'''


if args.cuda:
    train_net.cuda()

if args.opt == 'sgd':
    optimizer = optim.SGD(train_net.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=1e-4)
elif args.opt == 'adam':
    optimizer = optim.Adam(train_net.parameters(), lr=args.lr,
                           weight_decay=1e-4)
elif args.opt == 'rmsprop':
    optimizer = optim.RMSprop(train_net.parameters(), lr=args.lr,
                              momentum=0.9, weight_decay=1e-4)





def train(epoch):
    train_net.train()
    n_total_data = len(train_loader.dataset)
    n_batchs = len(train_loader)

    for batch_i, (inputs, target) in enumerate(train_loader):
        '''if len(inputs) < args.batch_size:
            continue'''

        time1 = time.time()
        if args.cuda:
            inputs, target = inputs.cuda(), target.cuda()
        inputs, target = Variable(inputs), Variable(target)

        optimizer.zero_grad()
        output = train_net(inputs)
        target = target.long()
        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()

        time2 = time.time()
        if batch_i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}\t'
                  'batch seconds: {:.2f}'.format(
                epoch, batch_i*args.batch_size, n_total_data,
                100.*batch_i/n_batchs, loss.data[0], time2-time1))

def test(epoch):
    train_net.eval()
    test_loss = 0
    correct = 0
    true_dict = {}
    err_dict = {}
    for i in range(100):
        true_dict[i] = [0,0]
        err_dict[i] = 0

    print('----------------')
    for inputs, target in test_loader:

        if args.cuda:
            inputs, target = inputs.cuda(), target.cuda()

        inputs, target = Variable(inputs), Variable(target)
        output = train_net(inputs)
        target = target.long()
        test_loss += float(F.nll_loss(output, target).data[0])

        output = output.cpu()
        target = target.cpu()
        pred = output.data.max(1)[1]

        #print(target)
        for y_p, y in zip(pred, target):
            if y_p == y:
                correct += 1
                true_dict[int(y)][0] += 1
            else:
                err_dict[int(y_p)] += 1
            true_dict[int(y)][1] += 1

        del inputs
        del target
        del output


    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    for i in range(100):
        acc = true_dict[i][0] / true_dict[i][1]
        true_dict[i] = acc * 100.0
    for k, v in true_dict.items():
        print(k,v)
    print('--------------')
    for k,v in err_dict.items():
        if v != 0:
            print(k,v)

if __name__ == '__main__':
    #deal_train()


    for epoch in range(1, args.epochs+1):
        train(epoch)
        torch.save(train_net.state_dict(), 'params.pkl')
        #test(epoch)


    #torch.save(train_net.state_dict(), 'params.pkl')

    '''train_net = resnext_101_32x4d
    train_net = nn.Sequential(*list(train_net.children())[:-1])
    to_add = nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x),
                           nn.Linear(2048, 100),
                           Lambda(lambda x: F.log_softmax(x)))

    train_net.add_module('fc', to_add)
    train_net.load_state_dict(torch.load('params.pkl'))
    train_net.eval()
    train_net.cuda()  
    test(0)'''
