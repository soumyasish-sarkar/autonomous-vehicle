import cv2
import numpy as np

img = cv2.imread(r"C:\Users\soumyasish-sarkar\PycharmProjects\opencv_learning\image\road1.jpg")
img=cv2.resize(img,(500,500))

cv2.imshow("image",img)

#Canny Edge detection Without applying GaussianBlur
#canny=cv2.Canny(img,50,150)
#cv2.imshow("Canny ED - without Gaussian Blur",canny)


img_g = cv2.GaussianBlur(img, (5,5), 0)
canny_g=cv2.Canny(img_g,50,150)
cv2.imshow("Canny ED - with Gaussian Blur",canny_g)

print(img.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()