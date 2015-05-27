import decimal

__author__ = 'ctizzle'

from skimage import io, color
import cv2
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math


def luv2lch(luv):
    import numpy as np
    import math

    lch = luv.copy()

    x, y, z = lch.shape

    for i in range (0, x, 1):
        for j in range(0, y, 1):
            u = luv[i][j][1]
            v = luv[i][j][2]
            lch[i][j][1] = math.sqrt(u**2 + v**2)
            lch[i][j][2] = math.atan2(v,u)


    return lch


img = mpimg.imread('Necromancer-icon-small.png')
print img.shape
print type(img)
print img

print img.size
print img.dtype

img4 = img/255
print img4

img1 = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

x, y, z = img1.shape

lmda = x * y
alpha = 1e0

lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)



lch = luv2lch(luv)

print lch

#plt.imshow(lch)
#plt.show()

theta = np.arctan2((luv[:, :, 2] - 0.48810), (luv[:, :, 1] - 0.20917))
qtheta = - 0.01585 - 0.03017 * np.cos(theta) - 0.04556 * np.cos(2 * theta)\
         - 0.02677 * np.cos(3 * theta) - 0.00295 * np.cos(4 * theta)\
         + 0.14592 * np.sin(theta) + 0.05084 * np.sin(2 * theta) \
         - 0.01900 * np.sin(3 * theta) - 0.00764 * np.sin(4 * theta)

#Saturation
suv = 13 * np.power(np.power((luv[:, :, 1] - 0.20917), 2) + np.power((luv[:, :, 2] - 0.48810), 2), 0.5)

LHK = luv[:, :, 0] + (-0.1340 * qtheta + 0.0872 * 0.8147419482) * suv * luv[:, :, 0]

labT = lab.T
print lab
print labT

deltalhk_x = np.roll(LHK, 0, -1) - np.roll(LHK, 0, 1)
deltalhk_y = np.roll(LHK, -1, 0) - np.roll(LHK, 1, 0)
deltalab_x = np.roll(lab, 0, -1) - np.roll(lab, 0, 1)
deltalab_y = np.roll(lab, -1, 0) - np.roll(lab, 1, 0)

deltal_x = deltalab_x[:, :, 0]
deltal_y = deltalab_y[:, :, 0]
deltalab3_x = np.power(deltalab_x[:, :, 0], 3) + np.power(deltalab_x[:, :, 1], 3) + np.power(deltalab_x[:, :, 2], 3)
deltalab3_y = np.power(deltalab_y[:, :, 0], 3) + np.power(deltalab_y[:, :, 1], 3) + np.power(deltalab_y[:, :, 2], 3)

if np.sign(deltalhk_x) == 0:
    if np.sign(deltal_x) == 0:
        signg_x = np.sign(deltalab3_x)
    else:
        signg_x = np.sign(deltal_x)
else:
    signg_x = np.sign(deltalhk_x)

if np.sign(deltalhk_y) == 0:
    if np.sign(deltal_y) == 0:
        signg_y = np.sign(deltalab3_y)
    else:
        signg_y = np.sign(deltal_y)
else:
    signg_y = np.sign(deltalhk_y)
    











