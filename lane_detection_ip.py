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
        #((3*width )// 4, height // 2),  # middle-right
        (width,height // 2 ),
        #(width // 4, height // 2)  # middle-left
        (0, height // 2 ),
    ]], np.int32)

#Fill ROI with white
cv2.fillPoly(mask, polygon, 255)
#Resultanted ROI
roi = cv2.bitwise_and(canny, mask)

#Hough Line Transformation to detect the straight line path
houghLine = cv2.HoughLinesP(
        roi,
        rho=1, #journal rho = 1
        theta=np.pi/180,
        threshold=150, #Journal threshold = 100 #at 50 noisy line getting marked
        minLineLength=100, #Journal minLineLength=100
        maxLineGap=50 #Journal maxLineGap=50
    )

#copy of canny
hough = np.copy(img)

#drawing hough lines on the image
# if houghLine is not None:
#     for line in houghLine:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(hough, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
if houghLine is not None:
    for line in houghLine:
        x1, y1, x2, y2 = line[0]

        # Calculate angle in degrees
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Filter: ignore near-horizontal lines
        if abs(angle) <5:  # threshold angle, adjust if needed
            continue
        cv2.line(hough, (x1, y1), (x2, y2), (0, 255, 0), 3)  # green lines

#displaying the images
cv2.imshow("image",img)
#cv2.imshow("Canny ED - with Gaussian Blur",canny)
cv2.imshow("ROI",roi)
cv2.imshow("Hough Line",hough)


cv2.waitKey(0)
cv2.destroyAllWindows()