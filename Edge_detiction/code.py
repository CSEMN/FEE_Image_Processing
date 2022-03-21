import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('img.png')#read Image
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_RGB =  cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img_RGB)
plt.title("Original IMG")
plt.show()

#Smoothing using Gaussian fillter for better output
gaussian = cv.GaussianBlur(img_gray,(3,3),0)

plt.imshow(gaussian,cmap='gray')
plt.title("Gaussian IMG")
plt.show()

# Sobel Edge Detector
sobelx = cv.Sobel(gaussian,cv.CV_8U,1,0,ksize=5)
sobely = cv.Sobel(gaussian,cv.CV_8U,0,1,ksize=5)
sobel = sobelx + sobely

plt.imshow(sobel,cmap='gray')
plt.title("Sobel IMG")
plt.show()

# Prewitt Gradient Operator
x_kernel = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
y_kernel = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewitt_x = cv.filter2D(gaussian, -1, x_kernel)
prewitt_y = cv.filter2D(gaussian, -1, y_kernel)
prewitt = prewitt_x + prewitt_y

plt.imshow(prewitt,cmap='gray')
plt.title("Prewitt IMG")
plt.show()

# Laplacian of Gaussian
laplacian = cv.Laplacian(gaussian,cv.CV_64F)
plt.imshow(laplacian,cmap='binary')
plt.title("Laplacian IMG")
plt.show()

canny = cv.Canny(gaussian,100,100)
plt.imshow(canny,cmap='gray')
plt.title("Canny IMG")
plt.show()