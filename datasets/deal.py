import os
import numpy as np
import pickle as pk
from PIL import Image
import time
import torch
import torch.nn as nn
import pretrain_model.resnext_101_32x4d as pre_res
from config import *
from datasets.imageag import *

#localpath = 'datasets/'
localpath = ''

def shuffle_data(x_arr, y_arr):
    index = np.arange(len(x_arr))
    np.random.shuffle(index)
    x_tmp = []
    y_tmp = []
    for i in index:
        x_tmp.append(x_arr[i])
        y_tmp.append(y_arr[i])
    x = np.array(x_tmp, dtype=np.float32)
    y = np.array(y_tmp, dtype=np.float32)
    return x, y

def deal_ag(img_path, filename):
    path = localpath+'re_ag/'
    func_arr = [en_lambda, random_crop, shift, randomColor,
                en_color, en_brightness, en_contrast,
                en_sharpness, randomGaussian, add_noise,
                rotate]


    filename = filename.split('.')[0]
    ag_path = [path+filename+str(i)+'.jpg'
                   for i in range(len(func_arr)+1)]

    img = Image.open(img_path)
    img.save(ag_path[0])
    for i, func in enumerate(func_arr):
        func(img, ag_path[i+1])

    return ag_path, len(ag_path)


def deal_resize():
    re_shape = (672, 224)
    def _resize(path,files):
        for filename in files:
            im = Image.open(path+'/'+filename)
            im = im.resize(re_shape, Image.ANTIALIAS)
            print(filename)
            im.save('re_'+path+'/'+filename)

    train_files = os.listdir('train')
    test_files = os.listdir('test')

    _resize('train', train_files)
    _resize('test', test_files)

def deal_add(img_path, filename):
    path = localpath+'re_ag/'

    func_arr = [random_crop, shift, randomGaussian,
                add_noise, rotate,  random_crop,
                shift, rotate, shift, rotate,
                random_crop, shift, rotate, add_noise,
                random_crop, shift, rotate, rotate]

    filename = filename.split('.')[0]
    ag_path = [path + filename + str(i + 1) + 'add' + '.jpg'
               for i in range(len(func_arr))]

    img = Image.open(img_path)
    for i, func in enumerate(func_arr):
        func(img, ag_path[i])

    return ag_path, len(ag_path)

def deal_train():
    add_label = [63,17,15,50,5,89,72,12,66,
                 41,34,70,16,35,84,39,48,46]
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x_all = []
    y_all = []
    data_dict = {}
    for i in range(100):
        data_dict[i] = []


    with open(localpath+'labels/train.txt', 'r') as f:
        for line in f.readlines():
            line = line.split(' ')
            filename = line[0]
            label = int(line[1]) - 1
            print(filename, label)

            data, num = deal_ag(localpath+'re_train/'+filename, filename)
            if (label+1) in add_label:
                data_add, num_add = deal_add(localpath+'re_train/'+filename, filename)
                data.extend(data_add)
                num += num_add

            data_dict[label].extend(data)
            x_all.extend(data)
            y_all.extend([label] * num)


            '''
            im = Image.open('train/'+filename)
            im = im.resize((224, 224), Image.ANTIALIAS)
            data = np.array(im, dtype=np.float32)

            data_dict[label].append(data)
            x_all.append(data)
            y_all.append(label)'''

    for k, v in data_dict.items():
        res_len = len(v) - 5
        x_train.extend(v[5:])
        y_train.extend([k]*res_len)
        x_test.extend(v[:5])
        y_test.extend([k]*5)
    print(len(x_all))

    with open('train.pkl', 'wb') as f:
        pk.dump([x_train,y_train,x_test,y_test], f)
    with open('all.pkl', 'wb') as f:
        pk.dump([x_all, y_all], f)


def deal_predict():
    files = os.listdir('test')
    print(files)
    for filename in files:
        im = Image.open('test/'+filename)
        im = im.resize((672, 224), Image.ANTIALIAS)
        im.save('re_test/'+filename)



def deal_resnext():
    # 预训练模型imagenet
    resnext = pre_res.resnext_101_32x4d
    resnext.load_state_dict(torch.load('../pretrain_model/resnext_101_32x4d.pth'))
    # 去掉7*7池化往后，输出shape[bs,2048,7,7]
    resnext = nn.Sequential(*list(resnext.children())[:-3])
    resnext.cuda()
    resnext.eval()

    with open('all.pkl', 'rb') as f:
        img_files, label = pk.load(f)

    i = 0
    inputs = []
    inputs_name = []
    for filename in img_files:
        img = Image.open(filename)
        data = np.array(img, dtype=np.float32)
        data = np.transpose(data, [2,0,1])
        inputs_name.append(filename)
        inputs.append(data)
        i += 1

        if i == 30:
            print(inputs_name)
            inputs = torch.Tensor(inputs)
            inputs = inputs.cuda()
            outputs = resnext(inputs)
            outputs = outputs.cpu()
            outputs = np.array(outputs.data)

            for x, name in zip(outputs, inputs_name):
                name = name[6:]
                name = name.split('.')[0]
                np.save('re_feature/'+name+'.npy',x)

            i = 0
            inputs = []
            inputs_name = []

    if inputs_name!=[] and inputs!=[]:
        inputs = torch.Tensor(inputs)
        inputs = inputs.cuda()
        outputs = resnext(inputs)
        outputs = outputs.cpu()
        outputs = np.array(outputs.data)

        for x, name in zip(outputs, inputs_name):
            name = name[6:]
            name = name.split('.')[0]
            np.save('re_feature/' + name + '.npy', x)



if __name__ == '__main__':
    #deal_resize()
    deal_train()
    #deal_predict()
    #deal_resnext()