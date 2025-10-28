import cv2
import numpy as np

def lane_detection_IP(frame):
    frame = cv2.resize(frame, (640, 360))

    # image -> gaussianBlur -> Canny -> ROI

    # converting image to greyscale
    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Applying GaussianBlur
    frame_gaussian = cv2.GaussianBlur(frame_grey, (5, 5), 0)

    # Adaptive canny thresholding
    # compute median pixel value
    median_pix = np.median(frame_gaussian)

    # set threashold based on median
    lower = int(max(0, 0.7 * median_pix))
    upper = int(min(255, 1.3 * median_pix))

    # Canny Edge detection  ##For adaptive threshold we won't use normal canny
    frame_canny = cv2.Canny(frame_gaussian, lower, upper)


    # ROI - Region of Interest
    # Gets the size of the input image (grayscale edge image), height = number of rows (y-axis), width = number of columns (x-axis)
    height, width = frame_canny.shape
    # Creates a black mask (all zeros) with the same size as the image, will be used to "cut out" the region you want to keep.
    mask = np.zeros_like(frame_canny)
    # ROI polygon Area
    polygon = np.array([[
        (0, height),  # bottom-left
        (width, height),  # bottom-right
        # ((3*width )// 4, height // 2),  # middle-right
        (width, height // 1.7),
        # (width // 4, height // 2)  # middle-left
        (0, height // 1.7),
    ]], np.int32)

    # Fill ROI with white
    cv2.fillPoly(mask, polygon, 255)
    # Resultanted ROI
    frame_roi = cv2.bitwise_and(frame_canny, mask)

    # Hough Line Transformation to detect the straight line path
    houghLine = cv2.HoughLinesP(
        frame_roi,
        rho=1,  # journal rho = 1
        theta=np.pi / 180,
        threshold=150,  # Journal threshold = 100 #at 50 noisy line getting marked
        minLineLength=100,  # Journal minLineLength=100
        maxLineGap=50  # Journal maxLineGap=50
    )

    # copy of canny
    frame_hough = np.copy(frame)

    # drawing hough lines on the image
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
            if abs(angle) < 5:  # threshold angle, adjust if needed
                continue
            cv2.line(frame_hough, (x1, y1), (x2, y2), (0, 255, 0), 3)  # green lines

    return frame_hough




img = cv2.imread(r"resources\Lane Detection Test Video 01.mp4")
#img = cv2.resize(img, (640, 360))
frame = lane_detection_IP(img)

cv2.imshow("Original image", img)
cv2.imshow("Lane detected image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()