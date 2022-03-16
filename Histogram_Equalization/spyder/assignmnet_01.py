# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 21:54:00 2022
@author: csemn
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def showHist(image,title):
    hist,bins = np.histogram(image.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(image.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.title(title)
    plt.show()
    
img = cv.imread('target.jpg',cv.IMREAD_GRAYSCALE)
showHist(img,"Original")
#Method 1 : OpenCv EqualizeHist
equ = cv.equalizeHist(img)
showHist(equ,"Equalizaed")
#Mehtod 2 : CLAHE
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
showHist(cl1,"CLAHE")
#show results
res = np.hstack((img,equ,cl1))
plt.figure(figsize=(10,10))
plt.title("ORIGINAL                              EQUALIZAED                              CLAHE")
plt.imshow(res,cmap='gray')
plt.show()
