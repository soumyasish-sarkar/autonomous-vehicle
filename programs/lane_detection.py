import cv2
import numpy as np

#------------------------------------
# sobel_edge_detection on L-channel ->  Gaussian Blur -> Magnitude Threshold -> Color Threshold (S + R channel)
#------------------------------------
def sobel_edge_detection(hls_frame, L, S):
    # Sobel Edge Detection on L-channel
    sobelx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)

    # Gradient magnitude
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))

    # Applying GaussianBlur
    blur = cv2.GaussianBlur(magnitude, (5, 5), 0)

    # Magnitude Threshold
    mag_binary = np.zeros_like(blur)
    mag_binary[(blur >= 50) & (blur <= 255)] = 255  # keep strong edges only

    # Color Threshold (S + R channel)
    R = hls_frame[:, :, 2]  # Red channel from original BGR frame

    s_binary = np.zeros_like(S)
    s_binary[(S >= 100) & (S <= 255)] = 255  # highlight saturated lane lines

    r_binary = np.zeros_like(R)
    r_binary[(R >= 150) & (R <= 255)] = 255  # highlight red/yellow lane paint

    color_combined = cv2.bitwise_or(s_binary, r_binary)

    # Combine Edge + Color Thresholds
    combined = cv2.bitwise_or(mag_binary, color_combined)

    # Convert to 3-channel image for display/output
    result = cv2.merge([combined, combined, combined])

    return result
#----------------x-------------------

#------------------------------------
# HLS function defination
#------------------------------------
def hls(frame):
    # convert BGR to HSL color space
    hls_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    # Extract BGR to HLS color space
    L = hls_frame[:, :, 1]  # channel index 1 = Lightness
    S = hls_frame[:, :, 2]  # channel index 2 = saturation

    # For visualization, we can stack the two channels side by side
    L_display = cv2.merge([L, L, L])  # Convert single channel to 3-channel grayscale
    S_display = cv2.merge([S, S, S])

    # Optional: Combine or return either channel for further processing
    #hls_frame = cv2.addWeighted(L_display, 0.5, S_display, 0.5, 0)

    return hls_frame, L , S

#----------------x-------------------

#------------------------------------
# Lane detection function defination
#------------------------------------
def lane_detection(frame):

    #hls operations
    processed_hls_frame, L, S = hls(frame)

    #sobel edge detection
    processed_frame = sobel_edge_detection(processed_hls_frame, L, S)


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

    out_frames =[]
    width = 640
    height = 360
    fps = int(frame_capture.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')


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

        # ----store processed frame ----
        out_frames.append(frame_processed)
        # ------------------------------

        # Display frame
        cv2.imshow("Original", raw_frame)
        cv2.imshow("Processed frame", frame_processed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    frame_capture.release()
    cv2.destroyAllWindows()

    # -----------------------------------
    # Ask for output file name *after* video ends
    # -----------------------------------
    file_name = input("\nEnter a name to save the processed video (without extension): ").strip()
    if file_name == "":
        file_name = "output_l_s"
    if not file_name.lower().endswith(".mp4"):
        file_name += ".mp4"

    out_path = f"outputs/{file_name}"
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Write all stored frames to the output file
    for frame in out_frames:
        out.write(frame)
    out.release()

#-------------x-------------

# Main Function Called
main()