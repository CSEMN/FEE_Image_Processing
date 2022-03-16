import cv2 as cv
#read image
img = cv.imread("lenna.png")

img.shape
img.max()
img.min()
#draw image
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.imshow(img)
plt.show()
#Default is BGR ... so we convert to RGB
goodColoredImg= cv.cvtColor(img,cv.COLOR_BGR2RGB)
plt.figure(figsize=(5,5))
plt.imshow(goodColoredImg)
plt.show()
#save image in JPG format
cv.imwrite("lenna.jpg",img)
#convert to grayscale image
grayImg= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
plt.figure(figsize=(5,5))
plt.imshow(grayImg,cmap='gray')
plt.show() 
cv.imwrite('lenna_gray.jpg', grayImg)

#read grayscale img
grysclImg = cv.imread('lenna.jpg',cv.IMREAD_GRAYSCALE)
plt.figure(figsize=(5,5))
plt.imshow(grysclImg,cmap='gray')
plt.show()
#split color channels
blue , green , red = img[:,:,0],img[:,:,1],img[:,:,2]
plt.figure(figsize=(5,5))
plt.imshow(blue,cmap='Blues')
plt.show()
#cut image
plt.figure(figsize=(5,5))
plt.imshow(goodColoredImg[0:256,125:512,:])
plt.show()
# remember to take a copy of the image and not manipulate the original.
lenna_red=img.copy()
lenna_red[:,:,0]=0 #blue
lenna_red[:,:,1]=0 #green
plt.imshow(cv.cvtColor(lenna_red,cv.COLOR_BGR2RGB))
plt.show()