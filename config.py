import argparse
import torch
import pickle as pk
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset



parser = argparse.ArgumentParser(description='pytorch minist')
#parser.add_argument('--data_path',type=str,default='datasets/resnext_train.pkl')
parser.add_argument('--data_path',type=str,default='datasets/train.pkl')
parser.add_argument('--no-cuda', action='store_true')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--log-interval', type=int, default=20)

#数据参数
parser.add_argument('--im-channels', type=int, default=3)
parser.add_argument('--im-row', type=int, default=224)
parser.add_argument('--im-col', type=int, default=224)
parser.add_argument('--n_pool', type=int, default=7)
parser.add_argument('--n-classes', type=int, default=100)

#训练参数
parser.add_argument('--batch-size', type=int, default=10)
parser.add_argument('--test-batch-size', type=int, default=10)
parser.add_argument('-epochs', type=int, default=15)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))

#创建参数
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

#设置随机数种子
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)



class TorchData(Dataset):
    def __init__(self, img_path, labels):
        self.img_path = img_path
        self.labels = labels

    def __getitem__(self, index):
        img = Image.open('datasets/'+self.img_path[index])
        #img = img.resize((672,224))
        data = np.array(img, dtype=np.float32)
        data = np.transpose(data, [2,0,1])

        '''name = self.img_path[index]
        name = name[6:]
        name = 'datasets/re_feature/' + name[:-3] + 'npy'
        data = np.load(name)'''


        target = self.labels[index]

        return data, target

    def __len__(self):
        return len(self.img_path)

def color_preprocessing(x):
    x = np.array(x)
    mean = [125.307, 122.95, 113.865]
    std = [62.9932, 62.0887, 66.7048]
    x = x.astype('float32')
    for i in range(3):
        x[:, :, i] = (x[:, :, i] - mean[i]) / std[i]
    return x

def gen_all_loader():
    with open('datasets/all.pkl', 'rb') as f:
        x_all, y_all = pk.load(f)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    all_loader = DataLoader(TorchData(x_all, y_all),
                              batch_size=args.batch_size, shuffle=True, **kwargs)
    return all_loader

def gen_loader():
    with open(args.data_path, 'rb') as f:
        data = pk.load(f)
        x_train, y_train = data[0], data[1]
        x_test, y_test = data[2], data[3]


    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = DataLoader(TorchData(x_train,y_train),
                             batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(TorchData(x_test,y_test),
                             batch_size=args.test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = gen_loader()
    print(len(train_loader.dataset))
    train_loader = gen_all_loader()
    print(len(train_loader.dataset))


