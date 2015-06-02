__author__ = 'Colleen Toth'

import skimage.color as color
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import matplotlib.cm as cm




def color_difference(lab_i, lab_j, luv_i, luv_j, alpha1):
    import numpy as np

    if alpha1 is None:
        alpha1 = 0.1
    delta_l = lab_i[0] - lab_j[0].copy()
    delta_l2 = ((lab_i[0] - lab_j[0])**2).copy()
    delta_a2 = ((lab_i[1] - lab_j[1])** 2).copy()
    delta_b2 = ((lab_i[2] - lab_j[2])** 2).copy()
    r = 3.59210244843    # 2.54 * 1.41421356237;
    g = np.power(np.add(delta_l2, np.power(np.divide(np.multiply(alpha1, np.power(np.add(delta_a2, delta_b2), 0.5)), r), 2)), .5)

    lhk_i = luv2lhk(luv_i[0], luv_i[1], luv_i[2])
    lhk_j = luv2lhk(luv_j[0], luv_j[1], luv_j[2])
    delta_lhk = lhk_i - lhk_j
    if delta_lhk == 0:
        if delta_l == 0:
            sign_g = np.sign(np.power(delta_l2, 1.5) + np.power(delta_a2, 1.5) + np.power(delta_b2, 1.5))
        else:
            sign_g = np.sign(delta_l)
    else:
        sign_g = np.sign(delta_lhk)

    g = sign_g * g

    return g


def luv2lhk(l,u,v):
    import numpy as np
    import math

    kbr = 0.2717 * (((6.469 + 6.362 * 20)**0.4495)/(6.469 + 20)**0.4495)
    suv = 13. * ((u - 0.20917) ** 2 + (v - 0.48810) ** 2) ** 0.5
    theta = math.atan((u - 0.20917) / (v - 0.44810))
    qtheta = - 0.01585 - 0.03017 * math.cos(theta) - 0.04556 * math.cos(2 * theta) - 0.02677 * math.cos(
        3 * theta) - 0.00295 * math.cos(4 * theta) + 0.14592 * math.sin(theta) + 0.05084 * math.sin(
        2 * theta) - 0.01900 * math.sin(3 * theta) - 0.00764 * math.sin(4 * theta)
    lhk = l + (-0.1340 * qtheta + 0.0872 * kbr) * suv * l

    return lhk

def luv2lch(luv):
    import numpy as np

    lch = luv.copy()

    x, y, z = lch.shape

    for n in range(x):
        for m in range(y):
            u = luv[n][m][1]
            v = luv[n][m][2]
            lch[n][m][1] = np.sqrt(np.power(u, 2) + np.power(v, 2))
            lch[n][m][2] = np.arctan2(v, u)

    return lch

img = cv2.imread('pimg.jpg')

if img.dtype != 'float32':
    img = cv2.normalize(img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)


img1 = img.copy()

x, y, z = img.shape

img1 = np.zeros((x, y, z))
img1 = img.copy()


lmd = x * y
alpha = None

#lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

lab = color.rgb2lab(img)
luv = color.rgb2luv(img)
LCH = color.lab2lch(img)

#LCH = luv2lch(luv)

Gx = np.zeros((x, y))
Gy = np.zeros((x, y))
for i in range(1, x-1):
    for j in range(1, y-1):
        Gx[i, j] = color_difference(lab[i + 1, j, :], lab[i - 1, j, :], luv[i + 1, j, :], luv[i - 1, j, :], alpha)
        Gy[i, j] = color_difference(lab[i, j + 1, :], lab[i, j - 1, :], luv[i, j + 1, :], luv[i, j - 1, :], alpha)

L = LCH[:, :, 0]
C = LCH[:, :, 1]
H = LCH[:, :, 2]
T = np.zeros((x, y, 9))

for i in range(x):
    for j in range(y):
        T[i, j, :] = np.multiply(C[i, j], [math.cos(H[i, j]), math.cos(2 * H[i, j]), math.cos(3 * H[i, j]), math.cos(4 * H[i, j]),
                                math.sin(H[i, j]), math.sin(2 * H[i, j]), math.sin(3 * H[i, j]), math.sin(4 * H[i, j]),
                                1])


U, V, Z = np.gradient(T)
Lx, Ly = np.gradient(L)

M_s = np.zeros((9, 9))
b_s = np.zeros((9, 1))


for i in range(1,x-1):
    for j in range(1, y-1):
        p = Gx[i, j] - Lx[i, j]
        q = Gy[i, j] - Ly[i, j]
        u = np.reshape(U[i, j, :], (9, -1))
        v = np.reshape(V[i, j, :], (9, -1))

        b_s = np.add(b_s, (np.multiply(p, u) + np.multiply(q, v)))
        M_s = M_s + (np.multiply(u, (np.reshape(u, (-1, 1)))) + np.multiply(v, (np.reshape(v, (-1, 1)))))


theta = (M_s + x * y * np.identity(9))
theta = np.linalg.lstsq(theta, b_s)[0]
newTheta = np.zeros((9, 1))
newTheta = np.reshape(theta, (9, 1))
tones = np.zeros((x, y))


for i in range(x):
    for j in range(y):
        f = T[i, j, :].reshape(1, 9, order='F')
        f = f * (1-newTheta)
        tones[i, j] = L[i, j] + np.multiply(f, C[i, j])[0][0]



print '\n\n'

plt.imshow(img1)
plt.show()
plt.imshow(tones, cmap=cm.Greys_r)
plt.show()















