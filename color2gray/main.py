__author__ = 'Colleen Toth'
# Implementation of "Robust Color-to-gray via Nonlinear Global Mapping" by Kim, et al, 2009 at Pohang
# University of Science and Technology (POSTECH) using Python 2.7

import skimage.color as color
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm
import scipy



# Calculates the color difference between surrounding Lab to Lab and Luv to Luv color pixels
def color_difference(lab_i, lab_j, luv_i, luv_j, alpha1):
    import numpy as np

    # sets alpha if not provided
    if alpha1 is None:
        alpha1 = 1.0
    # Difference between L in LAB for two given pixels
    delta_l = lab_i[0] - lab_j[0].copy()
    # Difference between L in LAB for two given pixels squared
    delta_l2 = ((lab_i[0] - lab_j[0])**2).copy()
    # Difference between A in LAB for two given pixels squared
    delta_a2 = ((lab_i[1] - lab_j[1])** 2).copy()
    # Difference between B in LAB for two given pixels squared
    delta_b2 = ((lab_i[2] - lab_j[2])** 2).copy()
    r = 3.59210244843  #2.54 * sqrt(2) from Kim, et al. paper
    # Color difference calculated by Eq(4) in Kim, et al paper
    g = np.power(np.add(delta_l2, np.power(np.divide(np.multiply(alpha1, np.power(np.add(delta_a2, delta_b2), 0.5)), r), 2)), .5)

    # Compute LHKi and j based on LUV values using luv2lhk function
    lhk_i = luv2lhk(luv_i[0], luv_i[1], luv_i[2])
    lhk_j = luv2lhk(luv_j[0], luv_j[1], luv_j[2])
    # difference between LHK i and j
    delta_lhk = lhk_i - lhk_j
    # compute sign for relative ordering of pixels. If sign(LHK) is 0, then use sign(delta_l),
    # if delta_l is 0, then use the sign(delta-l + delta_a + delta_b) as described in the Kim, et al. paper
    if delta_lhk == 0:
        if delta_l == 0:
            sign_g = np.sign(np.power(delta_l2, 1.5) + np.power(delta_a2, 1.5) + np.power(delta_b2, 1.5))
        else:
            sign_g = np.sign(delta_l)
    else:
        sign_g = np.sign(delta_lhk)

    g = sign_g * g

    return g


# Converts CIELUV colorspace to Helmholtz-Kohlrausch (H-K) lightness predictor from
# the 1997 Nayatani paper. The values for the following equations (and the equations
# themselves were obtained from "Simple Estimation Methods for the Helmholtz-Kohlrausch
# Effect" by Yoshinobu Nayatani.
def luv2lhk(l, u, v):
    import math

    # Coefficient for specifying the adaptive luminance dependency of the H-K effect Eq(5)
    kbr = 0.2717 * (((6.469 + 6.362 * 20)**0.4495)/(6.469 + 20)**0.4495)
    # Metric saturation of the test chromatic light with x,y Eq(6)
    suv = 13 * ((u - 0.20917) ** 2 + (v - 0.48810) ** 2) ** 0.5
    # Theta for q(theta) given by Eq(4)
    theta = math.atan((u - 0.20917) / (v - 0.44810))
    # Function for predicting the change of the H-K effect in different hues Eq(3)
    qtheta = - 0.01585 - 0.03017 * math.cos(theta) - 0.04556 * math.cos(2 * theta) - 0.02677 * math.cos(
        3 * theta) - 0.00295 * math.cos(4 * theta) + 0.14592 * math.sin(theta) + 0.05084 * math.sin(
        2 * theta) - 0.01900 * math.sin(3 * theta) - 0.00764 * math.sin(4 * theta)
    # Calculation for LHK from Kim, et al paper LHK = L + Lf(theta)S
    lhk = l + (-0.1340 * qtheta + 0.0872 * kbr) * suv * l

    return lhk


# read in image
img2 = cv2.imread('testColor.jpg', flags=cv2.IMREAD_COLOR)

img = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

# normalize image if not already in correct format
if img.dtype != 'float32':
    img = cv2.normalize(img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)

# get dimensions of input image
x, y, z = img.shape

# store a copy of original image for comparison
img1 = np.zeros((x, y, z))
img1 = img.copy()

alpha = None

# Convert to colorspaces CIELAB, CIELUV, and CIELCH
lab = color.rgb2lab(img)  # - RGB to LAB
luv = color.rgb2luv(img)  # - RGB to LUV
LCH = color.lab2lch(img)  # - LAB to LCH

# Compute G(x, y) for image
Gx = np.zeros((x, y))   # Create empty array for Gx component
Gy = np.zeros((x, y))   # Create empty array for Gy component
for i in range(1, x-1):
    for j in range(1, y-1):
        #Calculate the Gx and Gy per pixel using color difference from Eq 3 (on pg 2 of paper)
        Gx[i, j] = color_difference(lab[i + 1, j, :], lab[i - 1, j, :], luv[i + 1, j, :], luv[i - 1, j, :], alpha)
        Gy[i, j] = color_difference(lab[i, j + 1, :], lab[i, j - 1, :], luv[i, j + 1, :], luv[i, j - 1, :], alpha)

# Assign Lightness, chroma, and hue arrays from LCH to their own respective arrays
L = LCH[:, :, 0]
C = LCH[:, :, 1]
H = LCH[:, :, 2]
T = np.zeros((x, y, 9))

for i in range(x):
    for j in range(y):
        T[i, j, :] = np.multiply(C[i, j], [math.cos(H[i, j]), math.cos(2 * H[i, j]), math.cos(3 * H[i, j]), math.cos(4 * H[i, j]),
                                math.sin(H[i, j]), math.sin(2 * H[i, j]), math.sin(3 * H[i, j]), math.sin(4 * H[i, j]),
                                1])

# Calculate color gradient from T and L
U, V, Z = np.gradient(T)
Lx, Ly = np.gradient(L)

# Initialize matrices for M_s and b_s
M_s = np.zeros((9, 9))  # 9x9 matrix
b_s = np.zeros((9, 1))  # 9x1 vector

# Solve for energy function E_s
for i in range(1,x-1):
    for j in range(1, y-1):
        p = Gx[i, j] - Lx[i, j]
        q = Gy[i, j] - Ly[i, j]
        u = np.reshape(U[i, j, :], (9, -1))
        v = np.reshape(V[i, j, :], (9, -1))


        b_s = np.add(b_s, (np.multiply(p, u) + np.multiply(q, v)))
        M_s = M_s + (np.multiply(u, (np.reshape(u, (-1, 1)))) + np.multiply(v, (np.reshape(v, (-1, 1)))))

# computes x which minimizes M_s
X = (M_s + x * y * np.identity(9))
X = np.linalg.lstsq(X, b_s)[0]
newX = np.zeros((9, 1))
newX = np.reshape(X, (9, 1))

# initializes matrix for final image
final_image = np.zeros((x, y))


# Solve for g(x, y) = L + f(theta)C Eq(1), the energy function E_s (continued)
for i in range(x):
    for j in range(y):
        f = T[i, j, :].reshape(1, 9, order='F')
        f = f * (newX)
        final_image[i, j] = L[i, j] + np.multiply(f, C[i, j])[0][0]


# displays original and final image, saves grayscale conversion to .jpg
plt.imshow(img1)
plt.show()
plt.imshow(final_image, cmap=cm.Greys_r)
plt.show()
bw_img = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
#cv2.imshow('gray', bw_img)
#cv2.waitKey(0)
#scipy.misc.imsave('gray_7.png', bw_img)















