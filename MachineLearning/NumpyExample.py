# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 14:02:24 2014

Using numpy to simulate Random Walk

@author: BrightHush
"""

import numpy as np
import random

#simulate one random walk
position = 0
walk = [position]
steps = 1000
for i in xrange(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)
    
nsteps = 20
draws = np.random.randint(0, 2, size=nsteps)
print draws
steps = np.where(draws > 0, 1, -1)
print steps
walk = steps.cumsum()
print walk
print walk.min(), walk.max()

print (np.abs(walk) >= 3).argmax()

#simulate more than one random at one time
nwalks = 5
nsteps = 10
draws = np.random.randint(0, 2, size=(nwalks, nsteps))
steps = np.where(draws>0, 1, -1)
walks = steps.cumsum(1)

walks.max()
walks.min()

hits5 = (np.abs(walks) >= 5).any(1)
print type(hits5)

