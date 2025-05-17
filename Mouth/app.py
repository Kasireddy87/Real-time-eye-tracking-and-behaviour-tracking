from flask import Flask, render_template, Response
import cv2
import dlib
from scipy.spatial import distance

app = Flask(__name__)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/kasireddyshirisha/Desktop/Mini_project FIles/Mouth/shape_predictor_68_face_landmarks.dat")

def get_lip_distance(landmarks):
    upper_lip = landmarks[62]
    lower_lip = landmarks[66]
    return distance.euclidean((upper_lip.x, upper_lip.y), (lower_lip.x, lower_lip.y))

def generate_frames(threshold):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            lip_distance = get_lip_distance(landmarks.parts())
            if lip_distance > threshold:
                cv2.putText(frame, "Mouth Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Mouth Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

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
    # Adjust the threshold value as needed
    threshold = 15.0
    return Response(generate_frames(threshold), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
