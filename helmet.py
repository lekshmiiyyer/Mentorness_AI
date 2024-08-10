import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# Function to load YOLO model and classes
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    with open("helmet.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# Function to detect helmets in an image
def detect_helmet(img, net, output_layers, classes):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
#draw
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 1, color, 2)
    
    return img

# Streamlit app
def main():
    st.title("Helmet Detection App")
    st.sidebar.title("Choose Input Type")
    
    input_type = st.sidebar.selectbox("Select Input Type", ["Image", "Video"])
    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        if input_type == "Image":
            image = np.asarray(bytearray(file_bytes), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            st.image(detect_helmet(image, net, output_layers, classes), channels="BGR", caption="Processed Image")
        elif input_type == "Video":
            # Save video file to a temporary location
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(file_bytes)
            temp_file.close()

            # Read video from the temporary file
            cap = cv2.VideoCapture(temp_file.name)
            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = detect_helmet(frame, net, output_layers, classes)
                stframe.image(frame, channels="BGR")
            cap.release()

            # Delete the temporary file after processing
            os.remove(temp_file.name)

if __name__ == "__main__":
    net, classes, output_layers = load_yolo()
    main()
