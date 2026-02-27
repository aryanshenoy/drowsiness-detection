import cv2
import dlib
import joblib
import numpy as np
import pygame
import pyttsx3
import threading
import time
from scipy.spatial import distance

# Load models 
print("Loading models...")
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")
model     = joblib.load("assets/drowsiness_rf_126.pkl")
scaler    = joblib.load("assets/scaler_126.pkl")
print("Models loaded!")

# Audio setup 
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

def play_buzzer():
    sample_rate = 44100
    duration    = 0.8
    t    = np.linspace(0, duration, int(sample_rate * duration))
    wave = (np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)
    wave = np.column_stack([wave, wave])
    sound = pygame.sndarray.make_sound(wave)
    sound.play()

def speak(msg):
    def _run():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.setProperty('volume', 1.0)
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            engine.say(msg)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"Speech error: {e}")

    if not any(t.name == 'speech_thread' and t.is_alive() for t in threading.enumerate()):
        t = threading.Thread(target=_run, name='speech_thread', daemon=True)
        t.start()

# Helper functions 
def get_landmarks(gray, face):
    shape  = predictor(gray, face)
    points = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    return points

def convert_landmarks_to_features(points):
    flat = []
    for (x, y) in points:
        flat.append(x)
        flat.append(y)
    return np.array(flat).reshape(1, -1)

def calculate_ear(landmarks):
    def eye_ear(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
    left_eye  = [landmarks[i] for i in range(36, 42)]
    right_eye = [landmarks[i] for i in range(42, 48)]
    return (eye_ear(left_eye) + eye_ear(right_eye)) / 2.0

def calculate_mar(landmarks):
    A = distance.euclidean(landmarks[51], landmarks[59])
    B = distance.euclidean(landmarks[53], landmarks[57])
    C = distance.euclidean(landmarks[48], landmarks[54])
    return (A + B) / (2.0 * C)

def calculate_head_tilt(landmarks):
    left_eye_center  = np.mean([landmarks[i] for i in range(36, 42)], axis=0)
    right_eye_center = np.mean([landmarks[i] for i in range(42, 48)], axis=0)
    dx = right_eye_center[0] - left_eye_center[0]
    dy = right_eye_center[1] - left_eye_center[1]
    return np.degrees(np.arctan2(dy, dx))

def rule_based_override(prediction, ear, mar, tilt):
    """
    Override model prediction using hard rules.
    Catches cases the model misses.
    tilt can be negative (left) or positive (right) — use abs()
    """

    if ear < 0.25:
        return "Drowsy"

    if mar > 0.65:
        return "Drowsy"

    if abs(tilt) > 20:
        return "Drowsy"

    if abs(tilt) > 15 and ear < 0.30:
        return "Drowsy"

    if ear < 0.28 and mar > 0.55:
        return "Drowsy"

    return prediction 

# Alert messages 
ALERT_MESSAGES = [
    "Hey! You look drowsy. Please take a break!",
    "Wake up! Your eyes are closing.",
    "Attention! Drowsiness detected. Take a moment to rest.",
    "You seem very tired. Please stop and rest.",
    "Alert! Please stay awake. Your safety is important.",
    "You are falling asleep. Splash some water on your face!",
]

CONSECUTIVE_ALERT_THRESHOLD = 5
ALERT_COOLDOWN = 6
FACE_BUFFER_SIZE = 5
SKIP_FRAMES = 2

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

if not cap.isOpened():
    print("Could not open webcam. Try changing VideoCapture(0) to VideoCapture(1)")
    exit()

print("Drowsiness detection running... Press Q to quit.")

frame_count     = 0
alert_count     = 0
alert_msg_index = 0
last_alert_time = 0
face_buffer     = []
last_faces      = []

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Failed to read frame.")
        break

    frame = cv2.resize(frame, (640, 480))
    frame_count += 1

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    if frame_count % SKIP_FRAMES == 0:
        faces = detector(gray_eq, 1)
        if len(faces) == 0:
            faces = detector(rgb, 1)
        if len(faces) == 0:
            faces = detector(gray, 1)
        last_faces = faces
    else:
        faces = last_faces

    face_buffer.append(len(faces) > 0)
    if len(face_buffer) > FACE_BUFFER_SIZE:
        face_buffer.pop(0)
    face_visible = any(face_buffer)

    status = "No Face"
    color  = (200, 200, 200)

    if len(faces) > 0:
        face      = faces[0]
        landmarks = get_landmarks(gray, face)

        features = convert_landmarks_to_features(landmarks)[0].tolist()
        ear  = calculate_ear(landmarks)
        mar  = calculate_mar(landmarks)
        tilt = calculate_head_tilt(landmarks)
        features.extend([ear, mar, tilt])

        features        = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction      = model.predict(features_scaled)[0]

        prediction = rule_based_override(prediction, ear, mar, tilt)

        status = prediction
        color  = (0, 0, 255) if prediction == "Drowsy" else (0, 255, 0)
        alert_count = alert_count + 1 if prediction == "Drowsy" else 0

        #Trigger alert
        now = time.time()
        if alert_count >= CONSECUTIVE_ALERT_THRESHOLD and (now - last_alert_time > ALERT_COOLDOWN):
            play_buzzer()
            msg = ALERT_MESSAGES[alert_msg_index % len(ALERT_MESSAGES)]
            speak(msg)
            print(f"🔔 Alert: {msg}")
            alert_msg_index += 1
            last_alert_time  = now

        #Draw face box
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        #Draw overlays
        cv2.putText(frame, f"EAR: {ear:.2f}",          (10, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}",          (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Tilt: {tilt:.1f} deg",    (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Streak: {alert_count}",   (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    # Status bar
    if face_visible:
        cv2.putText(frame, f"Status: {status}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        cv2.putText(frame, "No Face Detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        alert_count = 0

    cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    #Drowsy alert banner
    if alert_count >= CONSECUTIVE_ALERT_THRESHOLD:
        cv2.putText(frame, "WAKE UP! DROWSY ALERT!", (30, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    cv2.imshow("Drowsiness Detection System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
print(f"Done. Total frames processed: {frame_count}")