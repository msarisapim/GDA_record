import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import re


def process_results(result):
    # Define colors for each label in BGR format
    colors = {0: (0, 0, 255),    # Red
              1: (255, 255, 100), # Blue
              2: (0, 255, 255),  # Green
              3: (147, 100, 200)}  # Pink

    # Copy the original image for cropping and annotating
    orig_image_for_cropping = result.orig_img.copy()
    image_for_drawing = result.orig_img.copy()

    # Dictionary to hold crops
    cropped_images = {}

    # Track drawn classes to ensure only one OBB per class
    drawn_classes = set()

    if result.obb.xyxyxyxy.numel() > 0:
        obbs = result.obb.xyxyxyxy.cpu().numpy()
        aabbs = result.obb.xyxy.cpu().numpy()
        classes = result.obb.cls.cpu().numpy()
        confidences = result.obb.conf.cpu().numpy()

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

def img2gray(img):
    img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def img2text(img):
    reader = easyocr.Reader(['th'])
    text_list = reader.readtext(img)
    text = ' '.join([result[1] for result in text_list]) # Extract text from each result tuple and join them into a single string
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

def browse_image():
    # Initialize Tkinter root
    root = tk.Tk()
    # Hide the main window
    root.withdraw()
    # Open file dialog and return the selected file path
    file_path = filedialog.askopenfilename()
    return file_path

def main():
    st.title("Nutritional Values Detector")

    # Load the model
    model = YOLO('best.pt')  # Adjust the path as necessary

    # File uploader allows the user to choose an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Convert the image from BGR to RGB
        image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display the uploaded image
        # st.image(image, caption='Uploaded Image')#,width=100)

        # Process the image
        results = model.predict(image)

        for result in results:
            crops, annotated_image = process_results(result)

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
            st.image(annotated_image, caption='Processed Image') #, width=100)

            # Display the nutritional values
            st.subheader("Nutritional Values:")
            st.write(f"Energy: {nutritional_values.get('Energy', 'N/A')} kcal")
            st.write(f"Sugar: {nutritional_values.get('Sugar', 'N/A')} g")
            st.write(f"Fat: {nutritional_values.get('Fat', 'N/A')} g")
            st.write(f"Sodium: {nutritional_values.get('Sodium', 'N/A')} mg")


if __name__ == "__main__":
    main()
