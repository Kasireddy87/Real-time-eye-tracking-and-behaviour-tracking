from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLO model
net = cv2.dnn.readNet("/Users/kasireddyshirisha/Desktop/Mini_project FIles/count/yolov3.weights", "/Users/kasireddyshirisha/Desktop/Mini_project FIles/count/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load COCO class labels
with open("/Users/kasireddyshirisha/Desktop/Mini_project FIles/count/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture (0 for the default camera)
cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return render_template('video.html')

def detect_people(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
            if confidence > 0.5 and class_id == classes.index("person"):
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    people_count = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            if label == "person":
                people_count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {int(confidence * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if people_count > 1:
        cv2.putText(frame, "Multiple people detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"People Count: {people_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    ret, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()

def gen():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)
        frame_bytes = detect_people(frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
