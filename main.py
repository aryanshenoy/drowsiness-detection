import cv2
import dlib
import joblib
import numpy as np
import pygame
import pyttsx3
import threading
import time
from scipy.spatial import distance
from collections import deque

#  LOAD MODELS
print("Loading models...")
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")
model     = joblib.load("assets/drowsiness_rf_126.pkl")
scaler    = joblib.load("assets/scaler_126.pkl")
print("Models loaded!")


#  AUDIO SETUP  
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


#  DETECTION HELPERS
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
    if ear < 0.25:             return "Drowsy"
    if mar > 0.65:             return "Drowsy"
    if abs(tilt) > 20:         return "Drowsy"
    if abs(tilt) > 15 and ear < 0.30:   return "Drowsy"
    if ear < 0.28 and mar > 0.55:       return "Drowsy"
    return prediction


#  ALERT MESSAGES  
ALERT_MESSAGES = [
    "Hey! You look drowsy. Please take a break!",
    "Wake up! Your eyes are closing.",
    "Attention! Drowsiness detected. Take a moment to rest.",
    "You seem very tired. Please stop and rest.",
    "Alert! Please stay awake. Your safety is important.",
    "You are falling asleep. Splash some water on your face!",
]


#  CONFIG  

CONSECUTIVE_ALERT_THRESHOLD = 5
ALERT_COOLDOWN  = 6
FACE_BUFFER_SIZE = 5
SKIP_FRAMES     = 2


#  UI PALETTE
BG_DARK    = (18, 20, 30)       # main background
PANEL_BG   = (25, 29, 45)       # right dashboard background
PANEL_EDGE = (45, 52, 80)       # border / divider lines
C_WHITE    = (230, 235, 248)
C_MUTED    = (95, 108, 140)
C_GREEN    = (0, 220, 110)
C_YELLOW   = (255, 210, 0)
C_RED      = (255, 55, 55)
C_CYAN     = (0, 200, 255)
C_ORANGE   = (255, 140, 0)


#  UI DRAWING UTILITIES

def put_text(img, text, pos, scale, color, thick=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, text, pos, font, scale, color, thick, cv2.LINE_AA)

def draw_rounded_rect(img, x1, y1, x2, y2, color, radius=10, thickness=-1):
    """Filled rounded rectangle."""
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(img, (cx, cy), radius, color, thickness)

def draw_metric_bar(img, x, y, label, value, lo, hi, warn, danger,
                    bar_w=220, bar_h=10):
    """
    Horizontal bar: green → yellow → red.
    label  : display name
    value  : current float value
    lo/hi  : expected min / max range for the bar fill
    warn   : yellow threshold
    danger : red threshold (high = bad when danger > warn, else reversed)
    """
    ratio = np.clip((value - lo) / (hi - lo), 0, 1)

    # colour logic: EAR is inverse (low = bad), MAR / tilt high = bad
    if danger < warn:      # EAR-style (lower = worse)
        if value < danger: bar_color = C_RED
        elif value < warn: bar_color = C_YELLOW
        else:              bar_color = C_GREEN
    else:                  # MAR / tilt style (higher = worse)
        if value > danger: bar_color = C_RED
        elif value > warn: bar_color = C_YELLOW
        else:              bar_color = C_GREEN

    # background track
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), (40, 46, 68), -1)
    # fill
    fill = int(ratio * bar_w)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + bar_h), bar_color, -1)
    # border
    cv2.rectangle(img, (x, y), (x + bar_w, y + bar_h), PANEL_EDGE, 1)

    # label on the left (above bar)
    put_text(img, label, (x, y - 6), 0.40, C_MUTED, 1)
    # value on the right
    put_text(img, f"{value:.2f}", (x + bar_w + 6, y + bar_h), 0.40, bar_color, 1)


def draw_fatigue_bar(img, x, y, alert_count, threshold, bar_w=220, bar_h=14):
    ratio = np.clip(alert_count / max(threshold, 1), 0, 1)
    segments = 10
    seg_w = (bar_w - segments + 1) // segments
    for i in range(segments):
        sx = x + i * (seg_w + 1)
        filled = ratio >= (i + 1) / segments
        if filled:
            shade = C_RED if ratio > 0.75 else (C_YELLOW if ratio > 0.40 else C_GREEN)
        else:
            shade = (35, 40, 60)
        cv2.rectangle(img, (sx, y), (sx + seg_w, y + bar_h), shade, -1)

    put_text(img, "FATIGUE LEVEL", (x, y - 6), 0.40, C_MUTED, 1)
    pct = int(ratio * 100)
    shade = C_RED if ratio > 0.75 else (C_YELLOW if ratio > 0.40 else C_GREEN)
    put_text(img, f"{pct}%", (x + bar_w + 6, y + bar_h), 0.40, shade, 1)


def draw_status_pill(img, x, y, status, flash_on):
    """Large status pill badge."""
    w, h = 200, 44
    if status == "Drowsy":
        bg_col  = C_RED   if flash_on else (140, 30, 30)
        fg_col  = C_WHITE
        label   = "\u26a0  DROWSY"
    elif status == "Alert":
        bg_col  = (0, 90, 50)
        fg_col  = C_GREEN
        label   = "\u2713  ALERT"
    else:
        bg_col  = (40, 44, 65)
        fg_col  = C_MUTED
        label   = "NO FACE"

    draw_rounded_rect(img, x, y, x + w, y + h, bg_col, radius=8)
    cv2.rectangle(img, (x, y), (x + w, y + h), fg_col, 1)   # thin border
    tw, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0], None
    put_text(img, label, (x + (w - tw[0]) // 2, y + 28), 0.65, fg_col, 2)


def draw_alert_log(img, x, y, log_entries, row_h=18):
    put_text(img, "ALERT LOG", (x, y - 6), 0.40, C_MUTED, 1)
    cv2.line(img, (x, y), (x + 240, y), PANEL_EDGE, 1)
    for i, entry in enumerate(log_entries):
        ey = y + 10 + i * row_h
        put_text(img, entry, (x, ey), 0.37, C_MUTED if i > 0 else C_ORANGE, 1)


def draw_divider(img, x, y, length, horizontal=True):
    if horizontal:
        cv2.line(img, (x, y), (x + length, y), PANEL_EDGE, 1)
    else:
        cv2.line(img, (x, y), (x, y + length), PANEL_EDGE, 1)


def fmt_time(seconds):
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"



#  WEBCAM

CAM_W, CAM_H  = 640, 480
PANEL_W       = 320
CANVAS_W      = CAM_W + PANEL_W
CANVAS_H      = CAM_H

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

if not cap.isOpened():
    print("Could not open webcam. Try VideoCapture(1)")
    exit()

print("Drowsiness detection running... Press Q to quit.")


#  STATE VARIABLES
frame_count      = 0
alert_count      = 0
alert_msg_index  = 0
last_alert_time  = 0
face_buffer      = []
last_faces       = []

session_start    = time.time()
total_drowsy_events = 0
alert_log        = deque(maxlen=5)   # rolling log of last 5 alerts

# for red flash overlay
flash_frames     = 0
FLASH_DURATION   = 12   # frames

# for status pill flash
flash_toggle     = False

ear_val  = 0.0
mar_val  = 0.0
tilt_val = 0.0
status   = "No Face"


#  MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to read frame.")
        break

    frame = cv2.resize(frame, (CAM_W, CAM_H))
    frame_count += 1

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    #  Face detection
    if frame_count % SKIP_FRAMES == 0:
        faces = detector(gray_eq, 1)
        if len(faces) == 0: faces = detector(rgb, 1)
        if len(faces) == 0: faces = detector(gray, 1)
        last_faces = faces
    else:
        faces = last_faces

    face_buffer.append(len(faces) > 0)
    if len(face_buffer) > FACE_BUFFER_SIZE:
        face_buffer.pop(0)
    face_visible = any(face_buffer)

    status = "No Face"
    color  = C_MUTED

    if len(faces) > 0:
        face      = faces[0]
        landmarks = get_landmarks(gray, face)

        features = convert_landmarks_to_features(landmarks)[0].tolist()
        ear_val  = calculate_ear(landmarks)
        mar_val  = calculate_mar(landmarks)
        tilt_val = calculate_head_tilt(landmarks)
        features.extend([ear_val, mar_val, tilt_val])

        features        = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction      = model.predict(features_scaled)[0]
        prediction      = rule_based_override(prediction, ear_val, mar_val, tilt_val)

        status = prediction
        color  = C_RED if prediction == "Drowsy" else C_GREEN
        alert_count = alert_count + 1 if prediction == "Drowsy" else 0

        #  Trigger alert 
        now = time.time()
        if alert_count >= CONSECUTIVE_ALERT_THRESHOLD and (now - last_alert_time > ALERT_COOLDOWN):
            play_buzzer()
            msg = ALERT_MESSAGES[alert_msg_index % len(ALERT_MESSAGES)]
            speak(msg)
            print(f"Alert: {msg}")
            alert_msg_index    += 1
            last_alert_time     = now
            total_drowsy_events += 1
            flash_frames        = FLASH_DURATION
            ts = time.strftime("%H:%M:%S")
            alert_log.appendleft(f"[{ts}]  Drowsy detected")

        #  Face box (style) 
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # corner accents
        ck = 14
        for (cx, cy, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (cx, cy), (cx + dx*ck, cy), color, 2)
            cv2.line(frame, (cx, cy), (cx, cy + dy*ck), color, 2)

    else:
        alert_count = 0

    #  Red flash overlay
    if flash_frames > 0:
        alpha = 0.25 * (flash_frames / FLASH_DURATION)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (CAM_W, CAM_H), (0, 0, 255), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        flash_frames -= 1

    #  Drowsy banner on camera feed 
    if alert_count >= CONSECUTIVE_ALERT_THRESHOLD:
        banner_y = CAM_H // 2
        flash_toggle = not flash_toggle
        if flash_toggle:
            bw, bh = 460, 50
            bx = (CAM_W - bw) // 2
            draw_rounded_rect(frame, bx, banner_y - 34, bx + bw, banner_y + 16,
                              (180, 0, 0), radius=8)
            put_text(frame, "  DROWSINESS DETECTED  ",
                     (bx + 18, banner_y), 0.75, C_WHITE, 2)

    #  Minimal cam overlays (just face box label)
    if face_visible and status != "No Face":
        label_col = C_RED if status == "Drowsy" else C_GREEN
        put_text(frame, status.upper(), (10, 30), 0.65, label_col, 2)

    #  BUILD CANVAS  (cam on left, dashboard on right)
    canvas = np.full((CANVAS_H, CANVAS_W, 3), BG_DARK, dtype=np.uint8)
    canvas[0:CAM_H, 0:CAM_W] = frame

    # Vertical divider
    cv2.rectangle(canvas, (CAM_W, 0), (CAM_W + 2, CANVAS_H), PANEL_EDGE, -1)

    # Panel background
    cv2.rectangle(canvas, (CAM_W + 2, 0), (CANVAS_W, CANVAS_H), PANEL_BG, -1)

    PX = CAM_W + 18   # panel content left margin
    PY = 22           # panel content top start

    #  Title 
    put_text(canvas, "DASH", (PX, PY + 14), 0.75, C_CYAN, 2,
             cv2.FONT_HERSHEY_DUPLEX)
    put_text(canvas, "BOARD", (PX + 75, PY + 14), 0.75, C_WHITE, 2,
             cv2.FONT_HERSHEY_DUPLEX)
    put_text(canvas, "Driver Monitoring System", (PX, PY + 30),
             0.33, C_MUTED, 1)

    draw_divider(canvas, PX, PY + 40, PANEL_W - 30)
    cy = PY + 58

    #  Status Pill 
    draw_status_pill(canvas, PX, cy, status, flash_toggle)
    cy += 58

    draw_divider(canvas, PX, cy, PANEL_W - 30)
    cy += 14

    #  Metrics header 
    put_text(canvas, "BIOMETRIC INDICATORS", (PX, cy), 0.36, C_MUTED, 1)
    cy += 14

    # EAR bar  (safe range 0.20–0.45, warn<0.30, danger<0.25)
    draw_metric_bar(canvas, PX, cy + 10, "Eye Aspect Ratio (EAR)",
                    ear_val, lo=0.15, hi=0.45, warn=0.30, danger=0.25,
                    bar_w=220)
    cy += 34

    # MAR bar  (safe range 0.0–0.9, warn>0.55, danger>0.65)
    draw_metric_bar(canvas, PX, cy + 10, "Mouth Aspect Ratio (MAR)",
                    mar_val, lo=0.0, hi=0.90, warn=0.55, danger=0.65,
                    bar_w=220)
    cy += 34

    # Tilt bar (abs value, safe<15, warn>15, danger>20)
    draw_metric_bar(canvas, PX, cy + 10, "Head Tilt (deg)",
                    abs(tilt_val), lo=0, hi=35, warn=15, danger=20,
                    bar_w=220)
    cy += 38

    draw_divider(canvas, PX, cy, PANEL_W - 30)
    cy += 14

    # Fatigue bar 
    draw_fatigue_bar(canvas, PX, cy + 10, alert_count,
                     CONSECUTIVE_ALERT_THRESHOLD, bar_w=220)
    cy += 42

    draw_divider(canvas, PX, cy, PANEL_W - 30)
    cy += 14

    # Session stats
    put_text(canvas, "SESSION STATS", (PX, cy), 0.36, C_MUTED, 1)
    cy += 18
    elapsed = time.time() - session_start
    put_text(canvas, f"  Duration   {fmt_time(elapsed)}", (PX, cy), 0.42, C_WHITE, 1)
    cy += 18
    put_text(canvas, f"  Alerts     {total_drowsy_events}", (PX, cy), 0.42, C_WHITE, 1)
    cy += 18
    if last_alert_time > 0:
        put_text(canvas, f"  Last Alert  {time.strftime('%H:%M:%S', time.localtime(last_alert_time))}",
                 (PX, cy), 0.42, C_ORANGE, 1)
    else:
        put_text(canvas, "  Last Alert  --:--:--", (PX, cy), 0.42, C_MUTED, 1)
    cy += 22

    draw_divider(canvas, PX, cy, PANEL_W - 30)
    cy += 12

    # Alert log 
    draw_alert_log(canvas, PX, cy + 6, list(alert_log))
    cy += 12 + 5 * 18

    #  Footer 
    footer_y = CANVAS_H - 14
    cv2.line(canvas, (PX, footer_y - 8), (CANVAS_W - 14, footer_y - 8), PANEL_EDGE, 1)
    put_text(canvas, f"Frame #{frame_count}   |   Press Q to quit",
             (PX, footer_y), 0.32, C_MUTED, 1)

    #  SHOW
    cv2.imshow("Driver Drowsiness Detection", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
print(f"\nSession ended. Total frames: {frame_count} | Drowsy events: {total_drowsy_events}")
