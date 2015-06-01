#coding=utf8

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy as np

rng = np.random.RandomState(23455)

input = T.tensor4(name='input')

w_shp = (2, 3, 9, 9)
w_bound = np.sqrt(3*9*9)
W = theano.shared(np.asarray(
        rng.uniform(low=-1.0/w_bound, high=1.0/w_bound, size=w_shp),
        dtype=input.dtype), name='W')

b_shp = (2, )
b = theano.shared(np.asarray(
        rng.uniform(low=-0.5, high=0.5, size=b_shp),
        dtype=input.dtype), name='b')

conv_out = conv.conv2d(input, W)

output = T.nnet.sigmoid(conv_out+b.dimshuffle('x', 0, 'x', 'x'))

f = theano.function([input], output)

import pylab
from PIL import Image

img = Image.open('E:\\Github\\MachineLearning\\3wolfmoon.jpg')
img = np.asarray(img, dtype='float32') / 256.0

ish = img.shape
img_ = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, 3, ish[0], ish[1])
filtered_img = f(img_)

pylab.subplot(1, 3, 1)
#pylab.axis('off')
pylab.imshow(img)
#pylab.gray()

pylab.subplot(1, 3, 2)
pylab.imshow(filtered_img[0, 0, :, :])

pylab.subplot(1, 3, 3)
pylab.imshow(filtered_img[0, 1, :, :])

