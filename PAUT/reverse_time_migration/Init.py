#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 04:52:42 2024

@author: nephilim
"""

import matplotlib.pyplot as plt
import numpy as np
import skimage
# Let's assume you have some data to plot

Dig=np.load('correlation_rtm_result.npy')
Dig=Dig[10:-10,10:-10]
# Dig=skimage.transform.resize(Dig,(50,140))
plt.figure()
plt.imshow(Dig)
plt.title('Click on the graph!')

# Number of input points you want
n_points = 300

# Timeout in seconds, set to None to disable timeout
timeout = 100000  

# Call ginput
points = plt.ginput(n_points, timeout=timeout)
plt.close()

# points will be a list of tuples (x, y)
print("You clicked:")
x=[]
y=[]
for i, point in enumerate(points):
    x.append(point[0])
    y.append(point[1])
    print(f"Point {i}: (x={point[0]}, y={point[1]})")
    
x=np.array(x)
y=np.array(y)

ind=np.argsort(x)
x=np.take_along_axis(x, ind, axis=0)
y=np.take_along_axis(y, ind, axis=0)

yy1=np.interp(np.arange(450), xp=x, fp=y)
yy1=yy1.astype('int')

data=np.ones_like(Dig)*7

for idx in range(Dig.shape[1]):
    data[:yy1[idx],idx] = 4

plt.imshow(data)
np.save('Init.npy',data)
# plt.figure()
# plt.imshow(Dig)
