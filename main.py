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

# ─── LOAD MODELS ────────────────────────────────────────────────────────────
print("Loading models...")
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("assets/shape_predictor_68_face_landmarks.dat")
model     = joblib.load("assets/drowsiness_rf_126.pkl")
scaler    = joblib.load("assets/scaler_126.pkl")
print("Models loaded!")

# ─── AUDIO ──────────────────────────────────────────────────────────────────
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

def play_buzzer():
    sr = 44100
    t  = np.linspace(0, 0.8, int(sr * 0.8))
    w  = (np.sin(2 * np.pi * 880 * t) * 32767).astype(np.int16)
    pygame.sndarray.make_sound(np.column_stack([w, w])).play()

def speak(msg):
    def _run():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.setProperty('volume', 1.0)
            for v in engine.getProperty('voices'):
                if 'female' in v.name.lower() or 'zira' in v.name.lower():
                    engine.setProperty('voice', v.id); break
            engine.say(msg); engine.runAndWait(); engine.stop()
        except Exception as e:
            print(f"Speech error: {e}")
    if not any(t.name == 'speech_thread' and t.is_alive() for t in threading.enumerate()):
        threading.Thread(target=_run, name='speech_thread', daemon=True).start()

# ─── DETECTION HELPERS ──────────────────────────────────────────────────────
def get_landmarks(gray, face):
    shape = predictor(gray, face)
    return [(shape.part(i).x, shape.part(i).y) for i in range(68)]

def landmarks_to_features(pts):
    return np.array([c for p in pts for c in p]).reshape(1, -1)

def ear(lm):
    def _e(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
    return (_e([lm[i] for i in range(36,42)]) + _e([lm[i] for i in range(42,48)])) / 2.0

def mar(lm):
    A = distance.euclidean(lm[51], lm[59])
    B = distance.euclidean(lm[53], lm[57])
    C = distance.euclidean(lm[48], lm[54])
    return (A + B) / (2.0 * C)

def head_tilt(lm):
    lc = np.mean([lm[i] for i in range(36,42)], axis=0)
    rc = np.mean([lm[i] for i in range(42,48)], axis=0)
    return np.degrees(np.arctan2(rc[1]-lc[1], rc[0]-lc[0]))

def rule_override(pred, e, m, t):
    if e < 0.25: return "Drowsy"
    if m > 0.65: return "Drowsy"
    if abs(t) > 20: return "Drowsy"
    if abs(t) > 15 and e < 0.30: return "Drowsy"
    if e < 0.28 and m > 0.55: return "Drowsy"
    return pred

# ─── CONFIG ─────────────────────────────────────────────────────────────────
ALERT_MESSAGES = [
    "Hey! You look drowsy. Please take a break!",
    "Wake up! Your eyes are closing.",
    "Attention! Drowsiness detected. Rest now.",
    "You seem very tired. Please stop and rest.",
    "Alert! Stay awake. Your safety matters.",
    "You are falling asleep. Splash water on your face!",
]
CONSECUTIVE_ALERT_THRESHOLD = 5
ALERT_COOLDOWN   = 6
FACE_BUFFER_SIZE = 5
SKIP_FRAMES      = 2
FLASH_DURATION   = 12

# ─── PALETTE ────────────────────────────────────────────────────────────────
BLACK      = (0,   0,   0)
BG         = (8,   10,  16)        # near-black background
OVERLAY_BG = (12,  14,  22, 210)   # semi-transparent panel (BGRA)
ACCENT     = (0,   210, 255)       # cyan accent
C_WHITE    = (230, 235, 248)
C_MUTED    = (80,  95,  130)
C_GREEN    = (0,   220, 110)
C_YELLOW   = (255, 210, 0)
C_RED      = (255, 55,  55)
C_ORANGE   = (255, 140, 0)
BORDER     = (35,  42,  70)

# ─── HELPERS ────────────────────────────────────────────────────────────────
def txt(img, t, pos, scale, color, thick=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(img, t, pos, font, scale, color, thick, cv2.LINE_AA)

def txt_center(img, t, cx, y, scale, color, thick=1):
    (w, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    txt(img, t, (cx - w//2, y), scale, color, thick)

def rrect(img, x1, y1, x2, y2, col, r=10, th=-1):
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), col, th)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), col, th)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(img, (cx,cy), r, col, th)

def overlay_rect(img, x1, y1, x2, y2, color_bgr, alpha=0.72):
    roi = img[y1:y2, x1:x2]
    patch = np.full_like(roi, color_bgr)
    cv2.addWeighted(patch, alpha, roi, 1-alpha, 0, roi)
    img[y1:y2, x1:x2] = roi

def bar(img, x, y, label, val, lo, hi, warn, danger, w=180, h=8):
    ratio = float(np.clip((val-lo)/(hi-lo), 0, 1))
    if danger < warn:
        col = C_RED if val < danger else (C_YELLOW if val < warn else C_GREEN)
    else:
        col = C_RED if val > danger else (C_YELLOW if val > warn else C_GREEN)
    cv2.rectangle(img, (x,y), (x+w, y+h), (30,36,55), -1)
    fill = int(ratio * w)
    if fill > 0: cv2.rectangle(img, (x,y), (x+fill, y+h), col, -1)
    cv2.rectangle(img, (x,y), (x+w, y+h), BORDER, 1)
    txt(img, label,          (x, y-5),      0.32, C_MUTED, 1)
    txt(img, f"{val:.2f}",   (x+w+5, y+h),  0.32, col,    1)

def segbar(img, x, y, val, maxv, w=180, h=10):
    ratio = float(np.clip(val/max(maxv,1), 0, 1))
    segs  = 10
    sw    = (w - segs + 1) // segs
    for i in range(segs):
        sx = x + i*(sw+1)
        filled = ratio >= (i+1)/segs
        if filled:
            c = C_RED if ratio > 0.75 else (C_YELLOW if ratio > 0.40 else C_GREEN)
        else:
            c = (25, 30, 48)
        cv2.rectangle(img, (sx,y), (sx+sw, y+h), c, -1)
    txt(img, "FATIGUE",          (x, y-5),      0.32, C_MUTED, 1)
    pct = int(ratio*100)
    c   = C_RED if ratio > 0.75 else (C_YELLOW if ratio > 0.40 else C_GREEN)
    txt(img, f"{pct}%",          (x+w+5, y+h),  0.32, c, 1)

def fmt_time(s):
    return f"{int(s)//60:02d}:{int(s)%60:02d}"

# ════════════════════════════════════════════════════════════════════════════
#  SPLASH SCREEN  (no camera needed)
# ════════════════════════════════════════════════════════════════════════════
WIN = "Driver Drowsiness Detection"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# get screen size from a dummy frame
_dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
cv2.imshow(WIN, _dummy)
cv2.waitKey(1)

# try to detect actual screen resolution
try:
    import ctypes
    user32 = ctypes.windll.user32
    SCR_W  = user32.GetSystemMetrics(0)
    SCR_H  = user32.GetSystemMetrics(1)
except Exception:
    SCR_W, SCR_H = 1280, 720

BTN_W, BTN_H = 280, 60
BTN_X = SCR_W//2 - BTN_W//2
BTN_Y = SCR_H//2 + 100
BTN_X2, BTN_Y2 = BTN_X + BTN_W, BTN_Y + BTN_H

pulse    = 0.0
start_requested = False

def on_mouse(event, x, y, flags, param):
    global start_requested
    if event == cv2.EVENT_LBUTTONDOWN:
        if BTN_X <= x <= BTN_X2 and BTN_Y <= y <= BTN_Y2:
            start_requested = True

cv2.setMouseCallback(WIN, on_mouse)

print("Showing splash screen...")
while not start_requested:
    splash = np.full((SCR_H, SCR_W, 3), BG, dtype=np.uint8)

    # subtle grid lines
    for gx in range(0, SCR_W, 60):
        cv2.line(splash, (gx,0), (gx, SCR_H), (14,17,28), 1)
    for gy in range(0, SCR_H, 60):
        cv2.line(splash, (0,gy), (SCR_W, gy), (14,17,28), 1)

    # # glowing center circle
    # pulse = (pulse + 0.04) % (2*np.pi)
    # glow_r = int(95 + 10 * np.sin(pulse))
    # for r in range(glow_r, glow_r-30, -3):
    #     alpha = (glow_r - r) / 30.0
    #     col   = tuple(int(c * alpha) for c in ACCENT)
    #     cv2.circle(splash, (SCR_W//2, SCR_H//2 - 70), r, col, 1)
    # cv2.circle(splash, (SCR_W//2, SCR_H//2 - 70), glow_r-28, (18,22,36), -1)

    # # eye icon inside circle
    # ex, ey = SCR_W//2, SCR_H//2 - 70
    # cv2.ellipse(splash, (ex,ey), (38,18), 0, 0, 360, ACCENT, 2)
    # cv2.circle(splash, (ex,ey), 10, ACCENT, -1)
    # cv2.circle(splash, (ex,ey),  4, BG,     -1)

    # title
    t1, t2 = "DROWSINESS", "GUARD"
    txt_center(splash, t1, SCR_W//2, SCR_H//2 + 10,  1.4, C_WHITE,  2)
    txt_center(splash, t2, SCR_W//2, SCR_H//2 + 55,  1.4, ACCENT,   2)
    txt_center(splash, "Real-time Driver Monitoring System",
               SCR_W//2, SCR_H//2 + 85, 0.50, C_MUTED, 1)
    
    # animated START button
    btn_pulse = int(8 * abs(np.sin(pulse)))
    btn_col   = tuple(min(255, c + btn_pulse*4) for c in (0, 160, 200))
    rrect(splash, BTN_X, BTN_Y, BTN_X2, BTN_Y2, btn_col, r=12)
    cv2.rectangle(splash, (BTN_X,BTN_Y), (BTN_X2,BTN_Y2), ACCENT, 2)
    txt_center(splash, "START DETECTING", SCR_W//2, BTN_Y + 38, 0.65, BG, 2)

    txt_center(splash, "or press  SPACE  to begin",
               SCR_W//2, BTN_Y2 + 28, 0.38, C_MUTED, 1)

    # version / footer
    txt(splash, "v2.0  |  Press Q to quit",
        (20, SCR_H-16), 0.32, C_MUTED, 1)

    cv2.imshow(WIN, splash)
    k = cv2.waitKey(30) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows()
        pygame.quit()
        exit()
    if k == ord(' '):
        start_requested = True

# ════════════════════════════════════════════════════════════════════════════
#  INIT CAMERA
# ════════════════════════════════════════════════════════════════════════════
CAM_W, CAM_H = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

if not cap.isOpened():
    print("Could not open webcam. Trying index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("No webcam found. Exiting.")
        cv2.destroyAllWindows()
        pygame.quit()
        exit()

print("Drowsiness detection active. Press Q to quit.")

# ─── STATE ──────────────────────────────────────────────────────────────────
frame_count         = 0
alert_count         = 0
alert_msg_index     = 0
last_alert_time     = 0
face_buffer         = []
last_faces          = []
session_start       = time.time()
total_drowsy_events = 0
alert_log           = deque(maxlen=4)
flash_frames        = 0
flash_toggle        = False
ear_val = mar_val = tilt_val = 0.0
status  = "No Face"

# Panel dimensions (right side overlay)
PANEL_W  = 280
PANEL_PAD = 14

# ════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════════════════════════════════
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Frame read failed.")
        break

    frame = cv2.resize(frame, (CAM_W, CAM_H))
    frame_count += 1

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    # ── Face detection ───────────────────────────────────────────────────
    if frame_count % SKIP_FRAMES == 0:
        faces = detector(gray_eq, 1)
        if not faces: faces = detector(rgb, 1)
        if not faces: faces = detector(gray, 1)
        last_faces = faces
    else:
        faces = last_faces

    face_buffer.append(len(faces) > 0)
    if len(face_buffer) > FACE_BUFFER_SIZE:
        face_buffer.pop(0)

    status = "No Face"

    if len(faces) > 0:
        face = faces[0]
        lm   = get_landmarks(gray, face)

        feats = landmarks_to_features(lm)[0].tolist()
        ear_val  = ear(lm)
        mar_val  = mar(lm)
        tilt_val = head_tilt(lm)
        feats.extend([ear_val, mar_val, tilt_val])

        feats_np  = np.array(feats).reshape(1, -1)
        pred      = model.predict(scaler.transform(feats_np))[0]
        pred      = rule_override(pred, ear_val, mar_val, tilt_val)
        status    = pred
        alert_count = alert_count + 1 if pred == "Drowsy" else 0

        # face box
        col = C_RED if pred == "Drowsy" else C_GREEN
        x1,y1,x2,y2 = face.left(),face.top(),face.right(),face.bottom()
        cv2.rectangle(frame, (x1,y1),(x2,y2), col, 2)
        ck = 14
        for (cx,cy,dx,dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame,(cx,cy),(cx+dx*ck,cy),col,2)
            cv2.line(frame,(cx,cy),(cx,cy+dy*ck),col,2)

        # alert trigger
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
            alert_log.appendleft(f"[{ts}] Drowsy event")

    else:
        alert_count = 0

    # ── Red flash overlay ────────────────────────────────────────────────
    if flash_frames > 0:
        alpha   = 0.30 * (flash_frames / FLASH_DURATION)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (CAM_W, CAM_H), (0,0,255), -1)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        flash_frames -= 1

    # ── Drowsy banner ────────────────────────────────────────────────────
    flash_toggle = not flash_toggle
    if alert_count >= CONSECUTIVE_ALERT_THRESHOLD and flash_toggle:
        bw, bh = 500, 52
        bx = (CAM_W - bw) // 2
        by = CAM_H // 2 - 26
        rrect(frame, bx, by, bx+bw, by+bh, (160,0,0), r=10)
        cv2.rectangle(frame, (bx,by),(bx+bw,by+bh), C_RED, 2)
        txt_center(frame, "⚠  DROWSINESS DETECTED  ⚠",
                   CAM_W//2, by+34, 0.80, C_WHITE, 2)

    # ════════════════════════════════════════════════════════════════════
    #  BUILD FULLSCREEN CANVAS
    # ════════════════════════════════════════════════════════════════════
    canvas = np.zeros((SCR_H, SCR_W, 3), dtype=np.uint8)

    # Letterbox / scale camera feed to fill height
    scale_f = SCR_H / CAM_H
    new_w   = int(CAM_W * scale_f)
    new_h   = SCR_H
    cam_scaled = cv2.resize(frame, (new_w, new_h))
    ox = (SCR_W - new_w) // 2         # horizontal offset if any
    canvas[:, max(0,ox):max(0,ox)+new_w] = cam_scaled[:, :min(new_w, SCR_W)]

    # ── RIGHT SIDE HUD PANEL ─────────────────────────────────────────────
    px1 = SCR_W - PANEL_W - 10
    px2 = SCR_W - 10
    py1 = 10
    py2 = SCR_H - 10

    overlay_rect(canvas, px1, py1, px2, py2, (8, 10, 18), alpha=0.82)
    cv2.rectangle(canvas, (px1,py1),(px2,py2), BORDER, 1)

    L  = px1 + PANEL_PAD       # left content x
    cy = py1 + 18              # cursor y

    # Title
    txt(canvas, "DROWSINESS", (L, cy), 0.58, ACCENT, 2, cv2.FONT_HERSHEY_DUPLEX)
    cy += 22
    txt(canvas, "GUARD",      (L, cy), 0.58, C_WHITE, 2, cv2.FONT_HERSHEY_DUPLEX)
    cy += 14
    cv2.line(canvas, (L, cy), (px2-PANEL_PAD, cy), BORDER, 1)
    cy += 16

    # Status pill
    pill_w = PANEL_W - 2*PANEL_PAD
    if status == "Drowsy":
        bg_c = C_RED if flash_toggle else (120,25,25)
        fg_c = C_WHITE
        lbl  = "DROWSY"
    elif status == "Alert":
        bg_c = (0, 72, 42)
        fg_c = C_GREEN
        lbl  = "ALERT"
    else:
        bg_c = (22, 26, 42)
        fg_c = C_MUTED
        lbl  = "NO FACE"

    rrect(canvas, L, cy, L+pill_w, cy+38, bg_c, r=8)
    cv2.rectangle(canvas, (L,cy),(L+pill_w,cy+38), fg_c, 1)
    txt_center(canvas, lbl, L + pill_w//2, cy+24, 0.68, fg_c, 2)
    cy += 48

    cv2.line(canvas, (L,cy),(px2-PANEL_PAD,cy), BORDER, 1)
    cy += 12

    # Biometrics
    txt(canvas, "BIOMETRICS", (L, cy), 0.33, C_MUTED, 1)
    cy += 14

    bar(canvas, L, cy, "EAR", ear_val, 0.15, 0.45, 0.30, 0.25, w=180)
    cy += 28
    bar(canvas, L, cy, "MAR", mar_val, 0.0,  0.90, 0.55, 0.65, w=180)
    cy += 28
    bar(canvas, L, cy, "TILT (deg)", abs(tilt_val), 0, 35, 15, 20, w=180)
    cy += 32

    cv2.line(canvas, (L,cy),(px2-PANEL_PAD,cy), BORDER, 1)
    cy += 12

    # Fatigue
    segbar(canvas, L, cy, alert_count, CONSECUTIVE_ALERT_THRESHOLD, w=180)
    cy += 30

    cv2.line(canvas, (L,cy),(px2-PANEL_PAD,cy), BORDER, 1)
    cy += 12

    # Session stats
    txt(canvas, "SESSION", (L, cy), 0.33, C_MUTED, 1)
    cy += 14
    elapsed = time.time() - session_start
    rows = [
        (f"Duration",  fmt_time(elapsed),              C_WHITE),
        (f"Alerts",    str(total_drowsy_events),        C_WHITE),
        (f"Last",      time.strftime('%H:%M:%S', time.localtime(last_alert_time))
                       if last_alert_time > 0 else "--:--:--", C_ORANGE),
        (f"Frames",    str(frame_count),                C_MUTED),
    ]
    for label, val, col in rows:
        txt(canvas, label, (L, cy),       0.36, C_MUTED, 1)
        txt(canvas, val,   (L+80, cy),    0.36, col,     1)
        cy += 18

    cv2.line(canvas, (L,cy),(px2-PANEL_PAD,cy), BORDER, 1)
    cy += 12

    # Alert log
    txt(canvas, "ALERT LOG", (L, cy), 0.33, C_MUTED, 1)
    cy += 14
    for i, entry in enumerate(alert_log):
        col_e = C_ORANGE if i == 0 else C_MUTED
        txt(canvas, entry, (L, cy), 0.30, col_e, 1)
        cy += 15

    # ── TOP BAR ──────────────────────────────────────────────────────────
    overlay_rect(canvas, 0, 0, SCR_W, 38, (8,10,18), alpha=0.70)
    cv2.line(canvas, (0,38),(SCR_W,38), BORDER, 1)

    ts_now = time.strftime("%H:%M:%S")
    txt(canvas, f"DROWSINESS GUARD  |  {ts_now}",
        (16, 24), 0.45, ACCENT, 1)

    hint = "Q - Quit"
    (hw,_),_ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    txt(canvas, hint, (SCR_W - hw - 16, 24), 0.40, C_MUTED, 1)

    # bottom-left: EAR/MAR/TILT quick readout
    bly = SCR_H - 20
    overlay_rect(canvas, 0, SCR_H-44, px1-10, SCR_H, (8,10,18), alpha=0.60)
    txt(canvas, f"EAR {ear_val:.2f}   MAR {mar_val:.2f}   TILT {abs(tilt_val):.1f}°   Session {fmt_time(elapsed)}",
        (16, bly), 0.40, C_MUTED, 1)

    cv2.imshow(WIN, canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
print(f"\nSession ended. Frames: {frame_count} | Drowsy events: {total_drowsy_events}")