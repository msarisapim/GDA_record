import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import re
import gdown
import os
import time

@st.cache_data
def process_results(_result, cache_invalidator):
    colors = {0: (0, 0, 255),    # Red
              1: (255, 255, 100), # Blue
              2: (0, 255, 255),  # Green
              3: (147, 100, 200)}  # Pink

    orig_image_for_cropping = _result.orig_img.copy()
    image_for_drawing = _result.orig_img.copy()
    cropped_images = {}
    drawn_classes = set()

    if _result.obb.xyxyxyxy.numel() > 0:
        obbs = _result.obb.xyxyxyxy.cpu().numpy()
        aabbs = _result.obb.xyxy.cpu().numpy()
        classes = _result.obb.cls.cpu().numpy()
        confidences = _result.obb.conf.cpu().numpy()

        for i, (obb, aabb, cls_id, conf) in enumerate(zip(obbs, aabbs, classes, confidences)):
            if conf >= 0.2 and cls_id not in drawn_classes:
                drawn_classes.add(cls_id)
                x1, y1, x2, y2 = map(int, aabb)
                if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= orig_image_for_cropping.shape[1] and y2 <= orig_image_for_cropping.shape[0]:
                    crop = orig_image_for_cropping[y1:y2, x1:x2]
                    if crop.size > 0:
                        cropped_images[cls_id] = crop
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
@st.cache_data
def img2gray(img):
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

@st.cache_data
def adjust_image_aspect_ratio(image, target_aspect_ratio=3/4):
    # Get the current dimensions of the image
    height, width = image.shape[:2]

    # Calculate the current aspect ratio
    current_aspect_ratio = width / height

    if current_aspect_ratio > target_aspect_ratio:
        # If the image is too wide, we crop the width
        new_width = int(target_aspect_ratio * height)
        start_x = (width - new_width) // 2
        cropped_image = image[:, start_x:start_x+new_width]
    else:
        # If the image is too tall, we crop the height
        new_height = int(width / target_aspect_ratio)
        start_y = (height - new_height) // 2
        cropped_image = image[start_y:start_y+new_height, :]

    return cropped_image


@st.cache_data
def img2text(img):
    reader = easyocr.Reader(['th'])
    text_list = reader.readtext(img)
    text = ' '.join([result[1] for result in text_list]) # Extract text from each result tuple and join them into a single string
    return text

@st.cache_data
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


def main():
    st.title("Nutritional Values Detector")

    # Offer the user the choice between using a camera or uploading an image
    choice = st.radio("Choose input method:", ("Use Camera", "Upload Image"))

    captured_image = None

    if choice == "Use Camera":
        captured_image = st.camera_input("Take a picture")
    elif choice == "Upload Image":
        captured_image = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])

    if captured_image is not None:
        # Convert the captured image to an OpenCV image
        file_bytes = np.asarray(bytearray(captured_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Adjust the aspect ratio of the image
        image = adjust_image_aspect_ratio(opencv_image)

        # Convert the image from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Model URL and path
        model_url = 'https://drive.google.com/uc?id=19kEKnJX-y_HOth28yiWn-xp1QTjajOJQ'
        model_path = 'best.pt'

        # Download and load the model
        download_model(model_url, model_path)
        model = YOLO(model_path)

        # Process the image
        results = model.predict(image)

        for result in results:
            # Generate a timestamp or other unique identifier as cache invalidator
            cache_invalidator = int(time.time())
            crops, annotated_image = process_results(_result=result, cache_invalidator=cache_invalidator)

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

            # Display the nutritional values with input fields for manual correction
            st.subheader("Nutritional Values:")
            energy = st.text_input("Energy (kcal):", value=str(nutritional_values.get('Energy', 'N/A')))
            sugar = st.text_input("Sugar (g):", value=str(nutritional_values.get('Sugar', 'N/A')))
            fat = st.text_input("Fat (g):", value=str(nutritional_values.get('Fat', 'N/A')))
            sodium = st.text_input("Sodium (mg):", value=str(nutritional_values.get('Sodium', 'N/A')))

            # Optionally, use the entered values for further processing or display
            if st.button('Save Nutritional Values'):
                # Save functionality or further processing can be added here
                st.success("Nutritional values have been saved/processed.")

if __name__ == "__main__":
    main()


