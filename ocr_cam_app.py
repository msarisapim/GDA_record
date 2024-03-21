import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import re
import gdown
import os
import hashlib

def generate_result_hash(result):
    """Generate a hash for the result object."""
    # Assuming result can be iterated to get detections with classes and confidences
    hash_str = ''.join(f"{det.cls}{det.conf}" for det in result.obb)
    return hashlib.md5(hash_str.encode()).hexdigest()

@st.cache_data
def process_results(_result, result_hash):
    # Define colors for each label in BGR format
    colors = {0: (0, 0, 255),    # Red
              1: (255, 255, 100), # Blue
              2: (0, 255, 255),  # Green
              3: (147, 100, 200)}  # Pink

    # Copy the original image for cropping and annotating
    orig_image_for_cropping = _result.orig_img.copy()
    image_for_drawing = _result.orig_img.copy()

    # Dictionary to hold crops
    cropped_images = {}

    # Track drawn classes to ensure only one OBB per class
    drawn_classes = set()

    if _result.obb.xyxyxyxy.numel() > 0:
        obbs = _result.obb.xyxyxyxy.cpu().numpy()
        aabbs = _result.obb.xyxy.cpu().numpy()
        classes = _result.obb.cls.cpu().numpy()
        confidences = _result.obb.conf.cpu().numpy()

        # Iterate over detections
        for i, (obb, aabb, cls_id, conf) in enumerate(zip(obbs, aabbs, classes, confidences)):
            if conf >= 0.2 and cls_id not in drawn_classes:
                # Mark the class as drawn
                drawn_classes.add(cls_id)

                x1, y1, x2, y2 = map(int, aabb)

                if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= orig_image_for_cropping.shape[1] and y2 <= orig_image_for_cropping.shape[0]:
                    crop = orig_image_for_cropping[y1:y2, x1:x2]
                    if crop.size > 0:
                        cropped_images[cls_id] = crop

                # Draw OBB on the original image
                color = colors.get(cls_id, (255, 255, 255))
                points = obb.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(image_for_drawing, [points], isClosed=True, color=color, thickness=2)

    return cropped_images, image_for_drawing

# def process_results(result):
#     # Define colors for each label in BGR format
#     colors = {0: (0, 0, 255),    # Red
#               1: (255, 255, 100), # Blue
#               2: (0, 255, 255),  # Green
#               3: (147, 100, 200)}  # Pink
#
#     # Copy the original image for cropping and annotating
#     orig_image_for_cropping = result.orig_img.copy()
#     image_for_drawing = result.orig_img.copy()
#
#     # Dictionary to hold crops
#     cropped_images = {}
#
#     # Track drawn classes to ensure only one OBB per class
#     drawn_classes = set()
#
#     if result.obb.xyxyxyxy.numel() > 0:
#         obbs = result.obb.xyxyxyxy.cpu().numpy()
#         aabbs = result.obb.xyxy.cpu().numpy()
#         classes = result.obb.cls.cpu().numpy()
#         confidences = result.obb.conf.cpu().numpy()
#
#         # Iterate over detections
#         for i, (obb, aabb, cls_id, conf) in enumerate(zip(obbs, aabbs, classes, confidences)):
#             if conf >= 0.2 and cls_id not in drawn_classes:
#                 # Mark the class as drawn
#                 drawn_classes.add(cls_id)
#
#                 x1, y1, x2, y2 = map(int, aabb)
#
#                 if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= orig_image_for_cropping.shape[1] and y2 <= orig_image_for_cropping.shape[0]:
#                     crop = orig_image_for_cropping[y1:y2, x1:x2]
#                     if crop.size > 0:
#                         cropped_images[cls_id] = crop
#
#                 # Draw OBB on the original image
#                 color = colors.get(cls_id, (255, 255, 255))
#                 points = obb.reshape((-1, 1, 2)).astype(np.int32)
#                 cv2.polylines(image_for_drawing, [points], isClosed=True, color=color, thickness=2)
#
#     return cropped_images, image_for_drawing

def img2gray(img):
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


@st.cache_data
def img2text(img):
    reader = easyocr.Reader(['th'])
    text_list = reader.readtext(img)
    text = ' '.join([result[1] for result in text_list])
    return text


def extract_1stnum(text):
    # Regular expression pattern for matching numbers
    pattern = r'\b\d+\b'  # This pattern matches whole numbers (no decimal points)
    # Use re.findall() to find all occurrences of the pattern
    numbers = re.findall(pattern, text)
    # Convert the extracted strings to integers
    numbers = [int(num) for num in numbers]

    # Check if any numbers were found
    if numbers:
        # Convert the extracted strings to integers and return the first one
        return int(numbers[0])
    else:
        # Return "N/A" if no numbers were found
        return "N/A"


@st.cache_resource
def download_model(url, output):
    """Download the model file from Google Drive."""
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model



def main():
    st.title("Nutritional Values Detector")

    # Model URL on Google Drive
    model_url = 'https://drive.google.com/uc?id=19kEKnJX-y_HOth28yiWn-xp1QTjajOJQ' #size L
    # model_url = 'https://drive.google.com/file/d/1Tjf1lVvHf6BMmazxZ0L7sa1rKRFHI9oR/view?usp=sharing' #size n
    model_path = 'best.pt'

    # Download the model if it doesn't exist
    download_model(model_url, model_path)

    # Load the model
    model = load_model(model_path)
    # model = YOLO('best.pt')  # Adjust the path as necessary

    # Use st.camera_input to capture an image from the webcam
    captured_image = st.camera_input("Take a picture or upload one")

    if captured_image is not None:
        # Convert the captured image to an OpenCV image
        file_bytes = np.asarray(bytearray(captured_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = model.predict(image)

        for result in results:
            # crops, annotated_image = process_results(result)
            result_hash = generate_result_hash(result)
            crops, annotated_image = process_results(result, result_hash)

            nutritional_values = {}

            for cls_id, crop in crops.items():
                crop_gray = img2gray(crop)
                text = img2text(crop_gray)
                val = extract_1stnum(text)

                if cls_id == 0:
                    nutritional_values['Energy'] = val
                elif cls_id == 1:
                    nutritional_values['Sugar'] = val
                elif cls_id == 2:
                    nutritional_values['Fat'] = val
                elif cls_id == 3:
                    nutritional_values['Sodium'] = val

            # Display the annotated image
            st.image(annotated_image, caption='Processed Image')

            # Display the nutritional values
            st.subheader("Nutritional Values:")
            st.write(f"Energy: {nutritional_values.get('Energy', 'N/A')} kcal")
            st.write(f"Sugar: {nutritional_values.get('Sugar', 'N/A')} g")
            st.write(f"Fat: {nutritional_values.get('Fat', 'N/A')} g")
            st.write(f"Sodium: {nutritional_values.get('Sodium', 'N/A')} mg")

if __name__ == "__main__":
    main()


