import cv2
import numpy as np

#------------------------------------
# Lane detection function defination
#------------------------------------
def lane_detection(frame):

    #convert BGR to HSL color space
    hls_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    # Extract BGR to HLS color space
    L = hls_frame[:, :, 1]  # channel index 1 = Lightness
    S = hls_frame[:, :, 2]  # channel index 2 = saturation

    # For visualization, we can stack the two channels side by side
    L_display = cv2.merge([L, L, L])  # Convert single channel to 3-channel grayscale
    S_display = cv2.merge([S, S, S])

    # Optional: Combine or return either channel for further processing
    hls_frame = cv2.addWeighted(L_display, 0.5, S_display, 0.5, 0)

    processed_frame = hls_frame


    return processed_frame

#------------------X-------------------


# -------------------------
# Main Function defination
# -------------------------
def main():

    #Read video frame
    frame_path = r"resources/Lane Detection Test Video 01.mp4"
    frame_capture = cv2.VideoCapture(frame_path) #webcam input cv2.VideoCapture(0)

    if not frame_capture.isOpened():
        print("Unable to open video frame")
        exit() #stop the code

    #-----------------------
    #Setup for video writer
    #-----------------------
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = "outputs/output_l_s.mp4"
    fps = int(frame_capture.get(cv2.CAP_PROP_FPS))
    width = 640
    height = 360
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))


    while True:
        ret, raw_frame = frame_capture.read()
        if not ret:
            break

        # Resize frame for consistency (optional)
        raw_frame = cv2.resize(raw_frame, (640, 360))

        #------------------------------
        #Called Lane detection function
        #------------------------------
        frame_processed = lane_detection(frame=raw_frame)
        #-------------x----------------

        # ----write processed frame ----
        out.write(frame_processed)
        # ------------------------------

        # Display frame
        cv2.imshow("Original", raw_frame)
        cv2.imshow("Processed frame -- l_s", frame_processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    out.release()
    frame_capture.release()
    cv2.destroyAllWindows()

#-------------x-------------


# Main Function Called
main()