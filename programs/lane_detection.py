import cv2
import numpy as np


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

    return hls_frame, L , S

#----------------x-------------------

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
# Static ROI - Region of Interest  --Needs to be updated to dynamically
#------------------------------------
def roi(processed_sed_frame):
    # ROI (Region of Interest)

    height, width, _ = processed_sed_frame.shape

    # Create a single-channel mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Define the ROI polygon (trapezoid shape)

    # have different data points in test code
    # polygon = np.array([[
    #     (0, height),
    #     (width, height),
    #     (int(width * 0.55), int(height * 0.6)),
    #     (int(width * 0.45), int(height * 0.6))
    # ]], np.int32)

    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width / 1.4), int(height / 1.5)),
        (int(width - width / 1.4), int(height / 1.5))
    ]], np.int32)

    # Fill the polygon area in the mask
    cv2.fillPoly(mask, polygon, 255)

    # Apply mask to the processed frame (keeps only ROI)
    processed_roi_frame = cv2.bitwise_and(processed_sed_frame, processed_sed_frame, mask=mask)
    return processed_roi_frame
#----------------x-------------------

#-------------------------------------
# Perspective Transformation
#-------------------------------------
def perspective(processed_roi_frame):
    height, width, _ = processed_roi_frame.shape

    # # Define source points (trapezoid in ROI)
    src = np.float32([
        [0, height],             # Bottom-left
        [width, height],         # Bottom-right
        [width / 1.4, height / 1.5],    # Top-right
        [width - width / 1.4, height / 1.5]     # Top-left
    ])

    # Define destination points (rectangle for bird's-eye view)
    dst = np.float32([
        [0, height],             # Bottom-left
        [width, height],         # Bottom-right
        [width, 0],              # Top-right
        [0, 0]                   # Top-left
    ])

    # src = np.float32([
    #     [width * 0.45, height * 0.65],
    #     [width * 0.55, height * 0.65],
    #     [width * 0.9, height],
    #     [width * 0.1, height]
    # ])
    # dst = np.float32([
    #     [0, 0],
    #     [width, 0],
    #     [width, height],
    #     [0, height]
    # ])

    # Get perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Apply warp perspective
    processed_pers_frame = cv2.warpPerspective(processed_roi_frame, M, (width, height))

    return processed_pers_frame, M, Minv

#-------------------------------------

#---------------------------------------
# Histogram Analysis (Find Lane Bases)
#---------------------------------------
def histogram(processed_pers_frame):
    #implement extra layer to convert again to GREY to ensure No error due to passing of BGR
    gray = cv2.cvtColor(processed_pers_frame, cv2.COLOR_BGR2GRAY)
    _, binary_frame = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    histogram = np.sum(binary_frame[binary_frame.shape[0]//2:,:],axis=0)
    midpoint = int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:])+midpoint


    return binary_frame, histogram, left_base, right_base
#---------------------------------------

#------------------------------------
# Sliding Window + Polynomial Fit
#------------------------------------
def sliding_window_polyfit(binary_frame, left_base, right_base):
    processed_slide_frame = np.dstack((binary_frame, binary_frame, binary_frame))
    nwindows = 9
    window_height = int(binary_frame.shape[0] / nwindows)
    nonzero = binary_frame.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = left_base
    rightx_current = right_base
    margin = 80
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_frame.shape[0] - (window + 1) * window_height
        win_y_high = binary_frame.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(processed_slide_frame, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(processed_slide_frame, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # left_fit = np.polyfit(lefty, leftx, 2)
    # right_fit = np.polyfit(righty, rightx, 2)

    if len(leftx) > 0 and len(lefty) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        left_fit = [0, 0, 0]
        print("⚠️ Warning: No left lane pixels found in this frame.")

    if len(rightx) > 0 and len(righty) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        right_fit = [0, 0, 0]
        print("⚠️ Warning: No right lane pixels found in this frame.")

    ploty = np.linspace(0, binary_frame.shape[0] - 1, binary_frame.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    processed_slide_frame[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    processed_slide_frame[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return processed_slide_frame, left_fit, right_fit, ploty, left_fitx, right_fitx

#------------------------------------

#------------------------------------
# Curvature and Center Offset
#------------------------------------
def measure_curvature_offset(left_fit, right_fit, binary_frame):
    ym_per_pix = 30 / binary_frame.shape[0]
    xm_per_pix = 3.7 / 700
    ploty = np.linspace(0, binary_frame.shape[0]-1, binary_frame.shape[0])
    y_eval = np.max(ploty)

    left_curverad = ((1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1])**2)**1.5) / abs(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval * ym_per_pix + right_fit[1])**2)**1.5) / abs(2 * right_fit[0])

    left_x = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_x = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_center = (left_x + right_x) / 2
    center_offset = ((binary_frame.shape[1]/2) - lane_center) * xm_per_pix * 100

    return left_curverad, right_curverad, center_offset

#------------------------------------

#------------------------------------
# Inverse Warp & Overlay
#------------------------------------
def overlay_lane(frame, binary_frame, left_fitx, right_fitx, ploty, Minv):
    color_warp = np.zeros_like(frame).astype(np.uint8)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int32([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (frame.shape[1], frame.shape[0]))
    result_frame = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)
    return result_frame

#--------------------------------------


#------------------------------------
# Lane detection function defination
#------------------------------------
def lane_detection(frame):

    #hls operations
    processed_hls_frame, L, S = hls(frame)

    #sobel edge detection
    processed_sed_frame = sobel_edge_detection(processed_hls_frame, L, S)

    #Static ROI
    processed_roi_frame = roi(processed_sed_frame)

    #Perspective Transformation
    processed_pers_frame, M, Minv = perspective(processed_roi_frame)

    #Histogram Analysis
    binary_frame, histogram_val, left_base, right_base = histogram(processed_pers_frame)

    #sliding window
    processed_slide_frame, left_fit, right_fit, ploty, left_fitx, right_fitx = sliding_window_polyfit(binary_frame, left_base, right_base)

    #measure curvature
    left_curv, right_curv, offset = measure_curvature_offset(left_fit, right_fit, binary_frame)

    result_frame = overlay_lane(frame, binary_frame, left_fitx, right_fitx, ploty, Minv)
    #cv2.putText(result_frame, f'Curvature: {(left_curv + right_curv) / 2:.1f}m', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 255, 255), 2)
    #cv2.putText(result_frame, f'Center Offset: {offset:+.1f} cm', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


    return result_frame


#------------------X-------------------


# -------------------------
# Main Function defination
# -------------------------
def main():

    #Read video frame
    frame_path = r"resources/Lane Detection Test Video 02b.mp4"
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