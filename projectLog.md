# Logs for my Autonomous Vehicle Project

## Tools to be installed

### Machine Learning - Image Processing Setup

1. **Anaconda / Miniconda**
   - Why: Helps create isolated environments (No "library conflict").
   - You can make one environment for ML (Tensorflow, Torch, Opencv), another later for ROS2.
   - **Recommended**: Miniconda (lighter than full Anaconda, necessary reqirements can be installed whenever its required).
      - Miniconda gives CLI no GUI, but GUI can be installed separately.
        > Download **Miniconda Installers** from 
        >> https://www.anaconda.com/download/success

      - `Leave all checkbox Unchecked during miniconda installation`


     - Create a new conda environment *To make preoject dependencies independent*
     - **Open Anaconda Prompt**
       > conda create -n `project_name` python=3.10

       > carla env list

       > conda activate `project_name`

   
2. **JupyterLab / Jupyter Notebook**
   - `Install inside conda itself`
   - Why: For ML training and for integration with CARLA/Gazebo, running ROS2 nodes, and testing how the trained model works in the simulation.
   - Easier to show demo during presentation (e.g., run a simulation and display detection results live).
   - **Recomended**: JupyterLab.
         
      > conda install -c conda-forge jupyterlab

      > jupyter-lab --version


3. **OpenCV**
   - `Install inside conda itself`
   - Why: Handle images & videos from camera dataset.
   - Even if you don’t use OpenCV algorithms heavily, it’s needed to read frames and preprocess images for ML. 

      > pip install opencv-python

4. **PyTorch**
   - `Install inside conda itself`
   - Why: YOLOv5/YOLOv8 depends on PyTorch.
   - **CPU-only version (no GPU needed)**
      - `cpuonly` version is sufficient for training/testing small datasets

         > conda install pytorch torchvision torchaudio cpuonly -c pytorch

   - **GPU version (if you have NVIDIA GPU & CUDA installed)**
      - Example for CUDA 11.8:

         > conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia


   - `Verify Installation`

      > python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

5. **YOLOv5 / YOLOv8**
   - Why: Pretrained models for pedestrian & traffic light detection.
   - No need to train from scratch. Later you can fine-tune if needed.

   - *YOLOv5 (classic):*
      > git clone https://github.com/ultralytics/yolov5 

      > cd yolov5

      > pip install -r requirements.txt

   - *YOLOv8 (latest, easier):*
      > pip install ultralytics




6. **Google Colab (Optional but Recommended)**
   - Why: For large GPU-intensive training for free.
   - Colab gives free Tesla T4 GPU, perfect for running YOLO on datasets.

   - `Run this on Colab terminal to store datasets and models to Google Drive.`
      > from google.colab import drive

      > drive.mount('/content/drive')


## Learning Requirements

## Image processing - Basics

- **Introduction** -> Overview of computer vision concepts.
- **Computer Vision Basics** -> Core principles and applications in autonomous vehicles.
- **OpenCV (Basic Functions)** -> Fundamental functions to read, display, and manipulate images.
- **Read and Show Image** -> `imread()`, `imshow()` usage.
- **Show Multiple Images** -> Displaying multiple frames simultaneously.
- **Draw Shapes on Image** -> Line, rectangle, circle, and ellipse drawing for visualization.
- **Text Over an Image** -> Overlaying text for annotations and debugging.

## Image processing - Advanced (Project centric)
> Frame Size used: (640,360) 

### Flow Chart for Lane detection
  - Image / Video -> (optional: Greyscale [nearly same result with or without] ) -> GaussianBlur [To remove noise from image] -> Canny Edge Detection -> Defining Region of Interest -> 


### Edge Detection -> Detect lane markings; basis for path planning.
  > Article for Lane detection :
  
  > https://ieeexplore.ieee.org/document/10499078

  >https://www.irjet.net/archives/V11/i3/IRJET-V11I3206.pdf

  **Edge Detection Techniques**
  - 1. **Canny Oerator** -> suppresses weak/noisy edges (via hysteresis), Produces thin edges (easy for Hough transform line detection), Tunable thresholds (50–150 can be adjusted based on road lighting).
  - 2. Sobel Operator -> Produces thicker edges, Good for detecting directional edges.
  - 3. Prewitt Operator (Similar to Sobel) -> Less accurate, rarely used in modern systems.
  - 4. Laplacian of Gaussian (LoG) -> More sensitive to noise, produces double edges.
  - 5. Deep Learning-based Edge Detectors -> Holistically-Nested Edge Detection (HED), RCF, DexiNed.
  - **Canny Edge Detection techniques** is used
    > canny=cv2.Canny(img,50,150)

**Defining Region of Interest**
- Define polygon (triangle/trapezoid focusing on road area)
- ROI to remove extra parts like sky, especially upper portion of the image/video
- We will use **Trapezium** / \ , because it gives wider view than traingle
- Lower base will be same as the lower verices of image itself, upper vertices of Trapezium somewhere around the center-top (near the horizon line).

### ROI Implementation
> height, width = img.shape
- Gets the size of the input image (grayscale edge image).
- height = number of rows (y-axis)
- width = number of columns (x-axis)
> mask = np.zeros_like(img)
- Creates a black mask (all zeros) with the same size as the image, will be used to "cut out" the region you want to keep.
>    polygon = np.array([[
        (0, height),      # bottom-left
        (width, height),  # bottom-right
        (width // 4, height // 2)  # middle-left
        (3*width // 4, height // 2)  # middle-right
    ]], np.int32)
- ROI polyg  >Area

> cv2.fillPoly(mask, polygon, 255)
- Fill ROI with white


> roi = cv2.bitwise_and(img, mask)
- Resultanted ROI

### Hough Line Detection
```python
houghLine = cv2.HoughLinesP(
        roi,
        rho=1, #journal rho = 1
        theta=np.pi/180,
        threshold=100, #Journal threshold = 100 #50 detect noisy lines too
        minLineLength=100, #Journal minLineLength=100
        maxLineGap=50 #Journal maxLineGap=50
    )
```
**NOTE :**  houghLine is not an image, it’s an array of line endpoints returned by cv2.HoughLinesP()
```python
# Draw detected lines - NOT ACCURATE
hough = np.copy(img)
if houghLine is not None:
    for line in houghLine:
        x1, y1, x2, y2 = line[0]
        cv2.line(hough, (x1, y1), (x2, y2), (0, 255, 0), 3)  # green lines
```
**NOTE :** Horizontal or near to horizontal lines are coming, we will remove those lines having angle < 20 degree with horizontal axis
```python
# Draw detected lines - NOT PERFECT FOR ROAD IS ON SHIFTED SKIGHTLY ONE SIDE
if houghLine is not None:
   for line in houghLine:
      x1, y1,x2,y2=line[0]

      # Calculate angle in degrees
      angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        
      # Filter: ignore near-horizontal lines
      if abs(angle) < 20:   # threshold angle, adjust if needed
         continue
      cv2.line(hough, (x1, y1), (x2, y2), (0, 255, 0), 3)  # green lines
```

> `img` (Input: binary edge image – output from Canny + ROI)
- White pixels = possible edges (where lines may exist).

> `rho=2`
- Distance resolution of the accumulator in pixels.
- Means we are checking for lines at steps of **2 pixels**.
- Smaller = more precise, but computationally heavier.

> `theta=np.pi/180`
- Angular resolution in radians.
- Here, it’s **1° (π/180 rad)**.
- Determines how finely we search for line angles.

> `threshold=50`
- Minimum number of votes in Hough accumulator to “confirm” a line.
- Higher → fewer, but stronger/more reliable lines.
- Lower → more lines detected (including noise).

> `minLineLength=40`
- Lines shorter than **40 pixels** are ignored.
- Useful to avoid detecting small specks or broken edges.

> `maxLineGap=20`
- Maximum gap (in pixels) between line segments that can still be connected.
- Helps merge broken lane markings into a single longer line.




## not done yet...

- **Scaling/Rotating Images** -> Data augmentation for ML training.
- **Image Blurring** -> Denoising to improve detection accuracy.
- **Play Video using OpenCV** -> Read and render video frames.
- **Capture Video from Camera** -> Real-time data collection.
- **Morphological Operations** -> Image cleaning for better feature extraction.
- **Extract Images from Video** -> Dataset generation for training.
- **Color Conversion** -> `cvtColor()` for preprocessing.
- **Region of Interest (ROI)** -> Focus on relevant image areas.
- **Flip, Rotate, Transpose** -> Data augmentation for robustness.
- **Color Spaces** -> Useful in traffic light detection (red/green segmentation).
- **Filter Color with OpenCV** -> Segmentation for detecting signals or objects.
- **Perspective Transformation** -> Optional, useful for lane-view transformations.
- **Thresholding** -> Simple thresholding and Otsu’s binarization.
- **Histogram Equalization** -> Improve night vision image quality.
- **Image Contours** -> Shape detection for preprocessing objects like pedestrians or stop signs.
- **Object Detection Using Contours** -> Real-time detection from webcam feed.
- **Hough Transformations** -> Line detection (lanes) and circle detection (traffic lights).
- **Vehicle Detection in Video Frames** -> Concepts transferable to pedestrian detection.
- **Pedestrian Detection from Streaming Video** -> Closest requirement to autonomous vehicle project.
