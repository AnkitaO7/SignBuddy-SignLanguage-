from flask import Flask, render_template, Response, request, jsonify
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
import time
import threading

app = Flask(__name__)

# -- Global state -------------------------------------------------------------
current_mode = "general"   # “general” or “number”
camera_on    = True        # whether Flask should show live frames or a blank frame

# -- Gesture detection setup -------------------------------------------------
detector    = HandDetector(maxHands=1)
offset      = 20
imgSize     = 224

# Load your two classifiers and label lists:
general_classifier = Classifier(
    r"Model\General\keras_model_retrained.h5",
    r"Model\General\labels_retrained.txt"
)
general_labels = ["Good", "Hello", "No", "Ok", "Thank You", "Yes"]

number_classifier = Classifier(
    r"Model\Number\keras_model.h5",
    r"Model\Number\labels.txt"
)
number_labels = ["Eight", "Five", "Four", "Nine", "One", "Seven", "Six", "Three", "Two", "Zero"]

# -- Flask routes -------------------------------------------------------------
@app.route('/set_mode/<mode>', methods=['POST'])
def set_mode(mode):
    global current_mode
    if mode in ("general", "number"):
        current_mode = mode
        return jsonify(success=True)
    return jsonify(success=False), 400

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_on
    camera_on = not camera_on
    status = 'on' if camera_on else 'off'
    return jsonify({'status': f'Camera turned {status}'})

@app.route('/video/<mode>')
def video_feed(mode):
    return Response(generate_frames(mode),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

# -- Frame generator ---------------------------------------------------------
def generate_frames(mode="general"):
    last_label = ""
    last_detect_time = 0
    spoken = False

    cap = cv2.VideoCapture(0)
    classifier = general_classifier if mode == "general" else number_classifier
    labels     = general_labels     if mode == "general" else number_labels

    try:
        while True:
            success, img = cap.read()
            if not success:
                break

            # If camera toggled off, send blank frame
            if not camera_on:
                blank = np.zeros((480, 640, 3), np.uint8)
                ret, buf = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       buf.tobytes() +
                       b'\r\n')
                continue

            imgOutput = img.copy()
            hands, _  = detector.findHands(img)

            if hands:
                x, y, w, h = hands[0]['bbox']
                y1 = max(0, y - offset)
                y2 = min(img.shape[0], y + h + offset)
                x1 = max(0, x - offset)
                x2 = min(img.shape[1], x + w + offset)

                imgCrop = img[y1:y2, x1:x2]
                if imgCrop.size != 0:
                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    aspect = h / w
                    if aspect > 1:
                        k      = imgSize / h
                        wCal   = math.ceil(k * w)
                        imgRes = cv2.resize(imgCrop, (wCal, imgSize))
                        gap    = (imgSize - wCal) // 2
                        imgWhite[:, gap:gap + wCal] = imgRes
                    else:
                        k      = imgSize / w
                        hCal   = math.ceil(k * h)
                        imgRes = cv2.resize(imgCrop, (imgSize, hCal))
                        gap    = (imgSize - hCal) // 2
                        imgWhite[gap:gap + hCal, :] = imgRes

                    _, idx       = classifier.getPrediction(imgWhite, draw=False)
                    current_label = labels[idx]

                    now = time.time()
                    if current_label != last_label:
                        last_label = current_label
                        last_detect_time = now
                        spoken = False
                    else:
                        if not spoken and now - last_detect_time >= 1:
                            def _speak(t):
                                engine = pyttsx3.init()
                                engine.say(t)
                                engine.runAndWait()
                            threading.Thread(target=_speak, args=(current_label,), daemon=True).start()
                            spoken = True

                    # Draw label
                    cv2.putText(imgOutput, current_label, (x, y - 30),
                                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                    cv2.rectangle(imgOutput, (x, y), (x + w, y + h),
                                  (0, 255, 0), 3)

            # Encode & yield
            ret, buf = cv2.imencode('.jpg', imgOutput)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buf.tobytes() +
                   b'\r\n')
    finally:
        cap.release()

# -- Start server ------------------------------------------------------------
if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5000
    print("\n  Flask starting up! Visit this URL in your browser:\n")
    print(f"    http://{host}:{port}/\n")
    app.run(host=host, port=port, debug=True)