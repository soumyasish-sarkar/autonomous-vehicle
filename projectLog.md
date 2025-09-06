# Logs for my Autonomous Vehicle Project

## Tools to be installed

1. **Anaconda / Miniconda**
   - Why: Helps create isolated environments (No "library conflict").
   - You can make one environment for ML (Tensorflow, Torch, Opencv), another later for ROS2.
   - **Recommended**: Miniconda (lighter than full Anaconda, necessary reqirements can be installed whenever its required).
      - Miniconda gives CLI no GUI, but GUI can be installed separately.
        > Download **Miniconda Installers** from 
        >> https://www.anaconda.com/download/success

     - Create a new conda environment *To make preoject dependencies independent*
       > conda create -n `project_name` python=3.10

       > conda activate `project_name`

   
2. **JupyterLab / Jupyter Notebook**
   - `Install inside conda itself`
   - Why: For ML training and for integration with CARLA/Gazebo, running ROS2 nodes, and testing how the trained model works in the simulation.
   - Easier to show demo during presentation (e.g., run a simulation and display detection results live).
   - **Recomended**: JupyterLab.
         
      > conda install -c conda-forge jupyterlab

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
      >> drive.mount('/content/drive')
