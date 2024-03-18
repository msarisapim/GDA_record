# Nutrition Recording System using OCR
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

--------

# Yolo OBB
reference: https://docs.ultralytics.com/datasets/obb/


# YOLO OBB Format
The YOLO OBB format uses four normalized corner points to define bounding boxes, structured as:
class_index, x1, y1, x2, y2, x3, y3, x4, y4


# Directory structure
```
- GDA-OBB
    ├─ images
    │   ├─ train
    │   └─ val
    └─ labels
        ├─ train_original
        └─ val_original

```
