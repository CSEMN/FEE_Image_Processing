# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 23:21:49 2022
@author: csemn
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread('img.png',cv.IMREAD_GRAYSCALE)
kernel = np.ones((5,5),np.uint8)
result = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.subplot(1,2,2)
plt.imshow(result, cmap='gray')
plt.title("Morphologic")
plt.show()