# Fine-tune baidu Image Dataset in Pytorch with ImageNet Pretrained Models

This repo provide an example for pytorh fine-tune in new image dataset. The codes contain CNN model, pytorch train code and some image augmentation methods.\
I use baidu competition's images, for more details  seeing http://dianshi.baidu.com/gemstone/competitions/detail?raceId=17.

## Pre-trained model

Pre-trian model is no limited, here I use resnext-101 params converted from torch model. For more detials seeing https://github.com/clcarwin/convert_torch_to_pytorch.


## Modify CNN

Here I just change 1000 fc layer into 100 fc layer. 


## Usage

1.convert torch model to pytorch model\
2.generate images by deal.py into a floder\
3.modify CNN to your own model\
4.train


