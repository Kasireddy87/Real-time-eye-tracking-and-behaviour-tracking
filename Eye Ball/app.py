from flask import Flask, render_template, Response, jsonify
import cv2
import dlib
import numpy as np

app = Flask(__name__)

# Initialize dlib's face detector (HOG-based) and create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor_path = "/Users/kasireddyshirisha/Desktop/Mini_project FIles/Eye Ball/shape_predictor_68_face_landmarks.dat"
try:
    predictor = dlib.shape_predictor(predictor_path)
except Exception as e:
    print(f"Error loading predictor: {e}")
    exit(1)

# Define indexes for the left and right eye
(lStart, lEnd) = (36, 42)
(rStart, rEnd) = (42, 48)

directions = {"left_eye": "N/A", "right_eye": "N/A"}

def get_eye_region(landmarks, eye_points, frame):
    points = np.array([landmarks[i] for i in eye_points], dtype=np.int32)
    rect = cv2.boundingRect(points)
    (x, y, w, h) = rect
    eye_region = frame[y:y+h, x:x+w]
    points = points - points.min(axis=0)
    return eye_region, points, (x, y, w, h)

def detect_eye_direction(eye_region, eye_points, eye_box):
    mask = np.zeros(eye_region.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [eye_points], 255)
    
    eye = cv2.bitwise_and(eye_region, eye_region, mask=mask)
    gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    
    _, thresh_eye = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = contours[0]
        
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            (x, y, w, h) = eye_box
            pupil_x = cX + x
            eye_center_x = x + w // 2
            
            # Check if the pupil is to the left or right of the center
            if pupil_x < eye_center_x:
                return "left"
            else:
                return "right"
    
    # If no contours or detection failed, return N/A or error handling
    return "N/A"

def generate_frames():
    global directions
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            for face in faces:
                landmarks = predictor(gray, face)
                landmarks = [(p.x, p.y) for p in landmarks.parts()]
                
                # Get left eye
                left_eye_region, left_eye_points, left_eye_box = get_eye_region(landmarks, range(lStart, lEnd), frame)
                
                # Get right eye
                right_eye_region, right_eye_points, right_eye_box = get_eye_region(landmarks, range(rStart, rEnd), frame)
                
                # Detect left and right eye directions
                left_direction = detect_eye_direction(left_eye_region, left_eye_points, left_eye_box)
                right_direction = detect_eye_direction(right_eye_region, right_eye_points, right_eye_box)
                
                # Store the directions
                directions["left_eye"] = left_direction
                directions["right_eye"] = right_direction
                print(f"Left Eye: {left_direction}, Right Eye: {right_direction}")  # Log the detected direction

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_directions')
def get_directions():
    return jsonify(directions)

if __name__ == '__main__':
    app.run(debug=True)
