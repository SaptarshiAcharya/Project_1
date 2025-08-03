from flask import Flask, render_template, Response, jsonify
import cv2
from detection.animal_detector import AnimalDetector
from detection.poaching_detector import detect_poaching_activity
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Simulate 6 feeds using the same webcam
CAMERAS = {
    "camera1": "http://192.168.165.64:8080/video",
    "camera2": 0,
    "camera3": 0,
    "camera4": 0,
    "camera5": 0,
    "camera6": 0
}


video_stream = cv2.VideoCapture(CAMERAS["camera2"])
detector = AnimalDetector("yolov8n.pt")

alert_enabled = True
human_detected = False  # <-- Track if a human is currently detected


def generate_frames():
    global alert_enabled, human_detected

    while True:
        success, frame = video_stream.read()
        if not success:
            break
        else:
            annotated_frame, detected_animals = detector.detect_animals(frame)
            poaching_alert, status = detect_poaching_activity(detected_animals)

            if poaching_alert and alert_enabled:
                status = "HUMAN DETECTED - ALERT ACTIVE"
                human_detected = True
            else:
                human_detected = False

            cv2.putText(annotated_frame, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255) if poaching_alert else (0, 255, 0), 2)

            annotated_frame = cv2.resize(annotated_frame, (500, 440))
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html', cameras=CAMERAS.keys())


@app.route('/video/<camera_name>')
def video(camera_name):
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle_alert', methods=['POST'])
def toggle_alert():
    global alert_enabled
    alert_enabled = not alert_enabled
    return jsonify({"alert_enabled": alert_enabled})


@app.route('/alert_status')
def alert_status():
    """Return whether a human is detected (used only for display, no sound)."""
    return jsonify({"human_detected": human_detected, "alert_enabled": alert_enabled})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)