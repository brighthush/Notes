# -*- coding: utf-8 -*-
"""
Created on Thu Dec 04 20:23:25 2014

@author: Bright Hush
"""

import numpy
import theano
import theano.tensor as T

rng = numpy.random
N = 20
feats = 2

x = T.matrix('x')
y = T.vector('y')

w = theano.shared(numpy.asarray([0.1, 0.2]), name='w')
b = theano.shared(0.1, name='b')

s = T.dot(x, w) + b

p = 1 / (1 + T.exp(-s))

y_predicted = (p > 0.5)

likehood = (y*T.log(p) + (1-y)*T.log(1-p)).mean()
gw, gb = T.grad(likehood, [w, b])

f = theano.function(inputs=[x, y], outputs=[p, y_predicted], 
                   updates=((w, w-0.1*gw), (b, b-0.1*gb)))
print f(numpy.asarray([[1, 3], [2, 4]]), [1, 0])