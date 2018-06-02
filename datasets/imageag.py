import numpy as np
import skimage
import random
from PIL import Image, ImageEnhance
from skimage.io import imread, imshow, imsave
from keras.preprocessing import image
from skimage import transform




#对每个像素点增强
def en_lambda(img,filepath):
    img = img.point(lambda i: i * 1.4)

    img.save(filepath)

#颜色
def en_color(img, filepath):
    factor = np.random.uniform(0,2)
    imgenhancer_Color = ImageEnhance.Color(img)
    img = imgenhancer_Color.enhance(factor)

    img = rotate(img, filepath)

#亮度
def en_brightness(img, filepath):
    factor = np.random.uniform(0.3, 1.2)
    imgenhancer_Brightness = ImageEnhance.Brightness(img)
    img = imgenhancer_Brightness.enhance(factor)

    rotate(img, filepath)

#对比度
def en_contrast(img, filepath):
    factor = np.random.uniform(0.3, 1.5)
    imgenhancer_Contrast = ImageEnhance.Contrast(img)
    img = imgenhancer_Contrast.enhance(factor)

    rotate(img, filepath)

#锐化
def en_sharpness(img, filepath):
    factor = np.random.uniform(0, 2)
    imgenhancer_Sharpness = ImageEnhance.Sharpness(img)
    img = imgenhancer_Sharpness.enhance(factor)

    rotate(img, filepath)

#随机裁剪
def random_crop(img, filepath):
    x = np.random.uniform(0, 72)
    y = np.random.uniform(0, 34)
    w = 600
    h = 190
    img = img.crop((x,y,x+w,y+h))
    img = img.resize((672, 224), Image.ANTIALIAS)
    img.save(filepath)

def randomColor(image, filepath):
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    img = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
    img.save(filepath)

#高斯去噪
def randomGaussian(image, filepath, mean=0.2, sigma=0.3):

    def gaussianNoisy(im, mean=0.2, sigma=0.3):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

        # 将图像转化成数组

    img = np.asarray(image)
    img.flags.writeable = True  # 将数组改为读写模式
    width, height = img.shape[:2]
    img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])
    img = Image.fromarray(np.uint8(img))
    img.save(filepath)

#加噪声
def add_noise(img, filepath):
    img = np.array(img)
    height, weight, channel = img.shape
    for i in range(5000):
        x = np.random.randint(0,height)
        y = np.random.randint(0,weight)
        img[x ,y ,:] = 255

    img = Image.fromarray(np.uint8(img))
    img.save(filepath)


#反转图片
def resever_img(img, filepath):
    img = np.array(img)
    img = img[::-1,:,:]
    img = Image.fromarray(np.uint8(img))
    img.save(filepath)

#随机旋转图片
def rotate(x, filepath, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    x = np.array(x)
    rotate_limit = (-10, 10)
    theta = np.pi / 180 * np.random.uniform(rotate_limit[0], rotate_limit[1])
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image.transform_matrix_offset_center(rotation_matrix, h, w)
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)

    img = Image.fromarray(np.uint8(x))
    img.save(filepath)

#随机平移
def shift(x, filepath, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    x = np.array(x)
    w_limit = (-0.08, 0.08)
    h_limit = (-0.15, 0.15)
    wshift = np.random.uniform(w_limit[0], w_limit[1])
    hshift = np.random.uniform(h_limit[0], h_limit[1])
    h, w = x.shape[row_axis], x.shape[col_axis] #读取图片的高和宽
    tx = hshift * h #高偏移大小，若不偏移可设为0，若向上偏移设为正数
    ty = wshift * w #宽偏移大小，若不偏移可设为0，若向左偏移设为正数

    translation_matrix = np.array([[1, 0, tx],
                                  [0, 1, ty],
                                  [0, 0, 1]])
    transform_matrix = translation_matrix
    x = image.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    img = Image.fromarray(np.uint8(x))
    img.save(filepath)


if __name__ == '__main__':
    #img = imread('re_train/1ad5ad6eddc451daea455ac1bcfd5266d016326c.jpg')
    img = Image.open('re_train/0b7b02087bf40ad1f6a8772a5c2c11dfa8ecceaf.jpg')

    rotate(img, 'a.jpg')