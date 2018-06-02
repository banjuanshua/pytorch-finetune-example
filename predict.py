import time
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from pretrain_model.resnext_101_32x4d import *
from torch.autograd import Variable
from densenet_fixed import DenseNet
from config import *


class TorchData(Dataset):
    def __init__(self, files):
        self.files = files

    def __getitem__(self, index):
        img = Image.open('datasets/re_test/'+self.files[index])
        data = np.array(img, dtype=np.float32)
        data = np.transpose(data, [2, 0, 1])
        return data

    def __len__(self):
        return len(self.files)



files = os.listdir('datasets/re_test')
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
predict_loader = DataLoader(TorchData(files),
                             batch_size=10, **kwargs)

#训练模型
train_net = resnext_101_32x4d
train_net = nn.Sequential(*list(train_net.children())[:-1])
to_add = nn.Sequential(Lambda(lambda x: x.view(1, -1) if 1 == len(x.size()) else x),
                       nn.Linear(30720, 100),
                       Lambda(lambda x: F.log_softmax(x)))

train_net.add_module('fc', to_add)
train_net.load_state_dict(torch.load('params.pkl'))
train_net.cuda()
train_net.eval()



def predict():
    pred_arr = []
    filename_arr = []
    for inputs in predict_loader:

        if args.cuda:
            inputs = inputs.cuda()
        inputs = Variable(inputs)
        output = train_net(inputs)
        pred = output.data.max(1)[1]

        print(pred)
        for p in pred:
            re = int(p.data) + 1
            pred_arr.append(re)

        del inputs
        del output
        del pred


    with open('predict.txt', 'w') as f:
        for name, label in zip(files, pred_arr):
            f.write(str(name)+' '+str(label))
            f.write('\n')




if __name__ == '__main__':
    predict()
