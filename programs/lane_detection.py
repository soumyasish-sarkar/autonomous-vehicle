import cv2
import numpy as np


def main():
    #Read video frame
    frame_capture = cv2.VideoCapture(r"resources/Lane Detection Test Video 01.mp4")

    #Show video
    while True:
        ret, frame = frame_capture.read()
        if not ret:
            break
        cv2.imshow("Original", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_capture.release()
    cv2.destroyAllWindows()

main()