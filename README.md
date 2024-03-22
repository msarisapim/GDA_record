# Nutrition Recording System using OCR


<p align="center">
  <img src="https://i.ibb.co/F4dgbCV/2567-03-20-15-03-48-Window.png" alt="Nutritional Values Detector" width="500"/>
</p>


Developed an innovative Nutrition Recording System aimed at streamlining dietary tracking by integrating YOLO (You Only Look Once) object detection with Optical Character Recognition (OCR). This system is designed to automatically identify and analyze nutritional information from food packaging, enabling users to effortlessly record and monitor their dietary intake. Here's the workflow of this project:

1. Training
- Convert YOLO 1.1 format to YOLO-OBB format
- Split dataset to train and val
- Write YAML file

2. YOLO-OBB
- Testing
- Crops and draw OBB
- Preprocessing before OCR
  - Import cropped images
  - Rotate/Skew

3.  OCR part
- Extract only 1st number

The experiments are shown in a notebook (GDA_yolo_OCR.ipynb)
--------

## Yolo OBB
reference: https://docs.ultralytics.com/datasets/obb/


### YOLO OBB Format
The YOLO OBB format uses four normalized corner points to define bounding boxes, structured as:
class_index, x1, y1, x2, y2, x3, y3, x4, y4


### Directory structure
```
- GDA-OBB
    ├─ images
    │   ├─ train
    │   └─ val
    └─ labels
        ├─ train_original
        └─ val_original

```
# Application

## Nutritional Values Detector

This repository contains the `ocr_app.py` script, which uses the Ultralytics YOLO model and EasyOCR to detect nutritional information on food packages from images. It's designed to extract key nutritional values such as energy (calories), sugar, fat, and sodium content.

### Prerequisites

Before running the script, ensure you have the following prerequisites installed:

- Python 3.8 or later
- OpenCV
- Ultralytics YOLO
- EasyOCR
- Streamlit

You can install the necessary Python libraries using pip:

```sh
pip install opencv-python-headless ultralytics easyocr streamlit
```

Note: Depending on your system, you might need opencv-python instead of opencv-python-headless.


### Model Download
The model best.pt used by the script is hosted on Google Drive. Download it using the following link:

[Download best.pt](https://drive.google.com/file/d/19kEKnJX-y_HOth28yiWn-xp1QTjajOJQ/view?usp=sharing)

After downloading, save the model file in the same directory as the ocr_app.py script or adjust the script to point to the location where you saved the model.

### Usage
To use the script, first clone this repository to your local machine:

```
git clone https://github.com/msarisapim/GDA_record
cd your/saved/path//GDA_record
```

Then, run the Streamlit application:
```sh
streamlit run ocr_cam_app.py
```
The Streamlit application will start, and you can navigate to the URL provided in your terminal to interact with the application.
or access my [Streamlit online gda-cam-app](https://gda-cam-app.streamlit.app/)

### Features
- Browse and select images of food packages.
- Detect and display key nutritional values: Energy (kcal), Sugar (g), Fat (g), and Sodium (mg).
- Visualize the detected areas on the food package image.

