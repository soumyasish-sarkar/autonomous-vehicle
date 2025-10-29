import cv2
import numpy as np


#------------------------------------
# Lane detection function defination
#------------------------------------
def lane_detection(frame):
    processed_frame= operations to be performed on frame #Need to be edited
    return processed_frame


#------------------X-------------------


# -------------------------
# Main Function defination
# -------------------------
def main():

    #Read video frame

    #webcam input use: cv2.VideoCapture(0)
    frame_path = r"resources/Lane Detection Test Video 01.mp4"
    frame_capture = cv2.VideoCapture(frame_path)

    if not frame_capture.isOpened():
        print("Unable to open video frame")
        exit() #stop the code


    while True:
        ret, raw_frame = frame_capture.read()
        if not ret:
            break

        # Resize frame for consistency (optional)
        raw_frame = cv2.resize(raw_frame, (640, 360))

        height, width = raw_frame.shape[:2] #frame.shape = (height, width, channel)

        #------------------------------
        #Called Lane detection function
        #------------------------------
        frame_processed = lane_detection(frame=raw_frame)
        #-------------x----------------


        # Display frame
        cv2.imshow("Original", raw_frame)
        cv2.imshow("Processed frame", frame_processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame_capture.release()
    cv2.destroyAllWindows()
#-------------x-------------


# Main Function Called
main()