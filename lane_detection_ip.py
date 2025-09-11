import cv2
import numpy as np

img = cv2.imread(r"C:\Users\soumyasish-sarkar\PycharmProjects\opencv_learning\image\road1.jpg")
img=cv2.resize(img,(640,360))

#image -> gaussianBlur -> Canny -> ROI

#Applying GaussianBlur
img_g = cv2.GaussianBlur(img, (5,5), 0)
#Canny Edge detection
canny=cv2.Canny(img_g,50,150)

#ROI - Region of Interest
#Gets the size of the input image (grayscale edge image), height = number of rows (y-axis), width = number of columns (x-axis)
height, width = canny.shape
#Creates a black mask (all zeros) with the same size as the image, will be used to "cut out" the region you want to keep.
mask = np.zeros_like(canny)
# ROI polygon Area
polygon = np.array([[
        (0, height),      # bottom-left
        (width, height),  # bottom-right
        ((3*width )// 4, height // 2),  # middle-right
        (width // 4, height // 2)  # middle-left
    ]], np.int32)

#Fill ROI with white
cv2.fillPoly(mask, polygon, 255)
#Resultanted ROI
roi = cv2.bitwise_and(canny, mask)

#displaying the images
cv2.imshow("image",img)
cv2.imshow("Canny ED - with Gaussian Blur",canny)
cv2.imshow("ROI",roi)


cv2.waitKey(0)
cv2.destroyAllWindows()