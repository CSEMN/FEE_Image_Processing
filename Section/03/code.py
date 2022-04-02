# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 15:14:12 2022
@author: csemn
"""
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def show_result(oldImg,newImg,oldTitle='Original',newTitle='New'):
    intensity_values=np.array([x for x in range(256)])
    plt.figure(figsize=(15,10))
    plt.subplot(2, 2, 1)
    plt.imshow(oldImg, cmap="gray")
    plt.title(oldTitle+" Image")
    plt.subplot(2, 2, 2)
    plt.bar(intensity_values, cv.calcHist([oldImg],[0],None,[256],[0,256])[:,0],width = 5)
    plt.title(oldTitle+" Hist")
    plt.xlabel('intensity')
    plt.subplot(2, 2, 3)
    plt.imshow(newImg, cmap="gray")
    plt.title(newTitle+" Image")
    plt.subplot(2, 2, 4)
    plt.bar(intensity_values, cv.calcHist([newImg],[0],None,[256],[0,256])[:,0],width = 5)
    plt.title(newTitle+" Hist")
    plt.xlabel('intensity')
    plt.tight_layout()
    plt.show()

'''
# Report-1:
# There are other nonlinear methods to improve contrast and brightness, these
  methods have different sets of parameters. In general, it’s difficult to
  manually adjust the contrast and brightness parameter, but there are algorithms 
  that improve contrast automatically.
'''
     
#Method 1: Histogram equalization
img = cv.imread("butterfly.jpg",cv.IMREAD_GRAYSCALE)
equalizedImg = cv.equalizeHist(img)
show_result(img, equalizedImg,newTitle="Equalizaed")

#Method 2:CLAHE
clahe = cv.createCLAHE(clipLimit=1.0)
cleaheImg = clahe.apply(img)
show_result(img, cleaheImg,newTitle="CLAHE")

'''
# Report-1 Conclusion:
# Equaliza hist function spreaded the histogram intensity all over the range.
# ClAHE function increased the contrast alittle bit which made the image clearer.
  for this image CLAHE is the best choice.
'''
#################################################################################
"""
# Report-2: cv2.THRESH_TRUNC and cv2.THRESH_OTSU
"""
threshold = 70
max_value = 255
min_value = 0

img = cv.imread("butterfly.jpg",cv.IMREAD_GRAYSCALE)
#Binary thresholding
binaryImg = cv.threshold(img,threshold,max_value,cv.THRESH_BINARY)[1]
show_result(img, binaryImg,newTitle="BINARY")
#Trunc thresholding
truncImg = cv.threshold(img,threshold,max_value,cv.THRESH_TRUNC)[1]
show_result(img, truncImg,newTitle="TRUNC")
#Otsu thresholding
otsuImg = cv.threshold(img,threshold,max_value,cv.THRESH_OTSU)[1]
show_result(img, otsuImg,newTitle="OTSU")

'''
# Report-2 Conclusion:
    # Binray : if the pixel value >= threshold then pixel = max value
                otherwise pixel = min value
    # Trunc  : if the pixel value >= threshold then pixel = max value
                otherwise pixel value do not change. 
    # OTSU   : choose the optimal threshold value.
'''
