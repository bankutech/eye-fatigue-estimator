import os
import cv2
import numpy as np
import mediapipe as mp
import time
import winsound
import collections
import sys
import threading
import torch
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pyre-ignore[21]
import config
from src.model import FatigueModel
from src.dataset import get_transforms

mp_face_mesh = mp.solutions.face_mesh

# Eye landmark indices for MediaPipe Face Mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# ── Thresholds & Settings ──
EAR_BLINK_THRESHOLD = 0.21       # EAR below this = eyes closed (blink)
BLINK_CONSEC_FRAMES = 3          # Min consecutive frames below threshold to count as 1 blink
DROWSY_TIME_THRESHOLD = 2.0      # Seconds with eyes closed to trigger drowsiness alarm
ALARM_FREQUENCY = 2500           # Alarm beep frequency (Hz)
ALARM_DURATION_MS = 500          # Alarm beep duration (ms)
ALARM_COOLDOWN = 3.0             # Seconds between alarm beeps

# ── Smoothing Settings ──
# We keep a rolling window of EAR values over the last ~1 second.
# Brief blinks (< 0.5s) are filtered out so they don't affect the fatigue score.
EAR_HISTORY_SIZE = 30            # ~1 second of frames at 30fps
BLINK_FILTER_FRAMES = 15         # Ignore EAR dips shorter than this many frames (~0.5s)


def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_ear(eye_landmarks, img_w, img_h):
    """Computes Eye Aspect Ratio (EAR) given eye landmarks."""
    pts = [(int(pt.x * img_w), int(pt.y * img_h)) for pt in eye_landmarks]

    vertical_1 = get_distance(pts[1], pts[5])
    vertical_2 = get_distance(pts[2], pts[4])
    horizontal = get_distance(pts[0], pts[3])

    if horizontal == 0:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def compute_smoothed_ear(ear_history):
    """
    Computes a smoothed EAR from the history buffer by filtering out
    brief blink dips. Normal blinks last ~3-8 frames. We replace those
    brief dips with the person's open-eye baseline so they don't
    affect the fatigue score.
    """
    if len(ear_history) == 0:
        return 0.3  # Default open-eye EAR

    values = list(ear_history)

    # Find the baseline (open-eye EAR) as the 75th percentile of recent values.
    # This is robust because even if someone blinks a lot, most frames
    # will have eyes open.
    baseline = float(np.percentile(values, 75))

    # Replace brief dips with baseline.
    # Walk through values: if a dip below threshold is shorter than
    # BLINK_FILTER_FRAMES, replace those frames with the baseline.
    filtered = values.copy()
    i = 0
    while i < len(filtered):
        if filtered[i] < EAR_BLINK_THRESHOLD:
            # Found start of a dip — find where it ends
            j = i
            while j < len(filtered) and filtered[j] < EAR_BLINK_THRESHOLD:
                j += 1
            dip_length = j - i
            # If this dip is short (a normal blink), replace it with baseline
            if dip_length < BLINK_FILTER_FRAMES:
                for k in range(i, j):
                    filtered[k] = baseline
            i = j
        else:
            i += 1

    # Return the mean of the filtered values
    return float(np.mean(filtered))


def draw_dashboard(frame, score, status_text, color, blink_count,
                   session_blinks, eyes_closed_duration, fps, avg_ear):
    """Draws a heads-up display (HUD) dashboard overlay on the frame."""
    h, w, _ = frame.shape

    # ── Semi-transparent dark panel at the top ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # ── Row 1: Fatigue Score + Status ──
    score_label = f"Fatigue: {score:.0f}/100"
    cv2.putText(frame, score_label, (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, status_text, (250, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ── Row 2: Blink Count + Eyes Closed Duration + FPS ──
    stats = (f"Blinks: {session_blinks}  |  "
             f"Eyes Closed: {eyes_closed_duration:.1f}s  |  "
             f"FPS: {fps:.0f}")
    cv2.putText(frame, stats, (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # ── Row 3: EAR value (debug info) ──
    ear_info = f"EAR: {avg_ear:.3f} (smoothed)"
    cv2.putText(frame, ear_info, (15, 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # ── Fatigue bar (bottom of panel) ──
    bar_x, bar_y, bar_w, bar_h = 15, 95, w - 30, 10
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill_w = int(bar_w * (score / 100.0))
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + fill_w, bar_y + bar_h), color, -1)


def run_inference():
    print("=" * 50)
    print("   EYE FATIGUE ESTIMATOR - Live Dashboard")
    print("=" * 50)
    print("Features:")
    print("  - Smoothed fatigue score (ignores brief blinks)")
    print("  - Blink counter")
    print("  - Drowsiness alarm (eyes closed > 2s)")
    print("  - Color-coded status (Green/Yellow/Red)")
    print()
    print("How it works:")
    print("  - Normal blinks are filtered out (won't affect score)")
    print("  - Score only rises when eyes stay closed/droopy")
    print("  - Alarm sounds if eyes stay shut for 2+ seconds")
    print()
    print("Press 'q' to quit.")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # ── Load PyTorch Model ──
    device = torch.device("cpu")
    best_model_path = os.path.join(config.MODELS_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        print("Warning: best_model.pth not found. Please train the model first.")
        return
        
    model = FatigueModel(pretrained=False).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    print("Loaded PyTorch model for inference.")
    val_transforms = get_transforms(is_train=False)

    # ── Tracking State ──
    blink_counter = 0
    session_blinks = 0
    eyes_closed_start = None
    eyes_closed_duration = 0.0
    last_alarm_time = 0.0
    prev_time = time.time()
    fps = 0.0

    # ── Setup Window ──
    cv2.namedWindow("Eye Fatigue Estimator", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Eye Fatigue Estimator", 1280, 720)
    cv2.setWindowProperty("Eye Fatigue Estimator", cv2.WND_PROP_TOPMOST, 1)

    # ── EAR Smoothing Buffer ──
    ear_history = collections.deque(maxlen=EAR_HISTORY_SIZE)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            fps = 1.0 / max(current_time - prev_time, 0.001)
            prev_time = current_time

            img_h, img_w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(rgb_frame)

            score = 0.0
            status_text = "No Face"
            color = (128, 128, 128)
            smoothed_ear = 0.3

            # pyre-ignore[16]
            if results.multi_face_landmarks:
                # pyre-ignore[16]
                face_landmarks = results.multi_face_landmarks[0]

                # ── Compute instantaneous EAR ──
                left_eye_pts = [face_landmarks.landmark[i] for i in LEFT_EYE]
                right_eye_pts = [face_landmarks.landmark[i] for i in RIGHT_EYE]

                left_ear = compute_ear(left_eye_pts, img_w, img_h)
                right_ear = compute_ear(right_eye_pts, img_w, img_h)
                avg_ear = (left_ear + right_ear) / 2.0

                # ── Add to history buffer ──
                ear_history.append(avg_ear)

                # ── Compute SMOOTHED EAR (filters out brief blinks) ──
                smoothed_ear = compute_smoothed_ear(ear_history)

                # ── Fatigue Score from PyTorch Model ──
                x_coords = [lm.x for lm in face_landmarks.landmark]
                y_coords = [lm.y for lm in face_landmarks.landmark]
                
                # Bounding box for crop (using 0.1 padding to match training data)
                x_min_raw = int(min(x_coords) * img_w)
                x_max_raw = int(max(x_coords) * img_w)
                y_min_raw = int(min(y_coords) * img_h)
                y_max_raw = int(max(y_coords) * img_h)
                
                pad_x_crop = int((x_max_raw - x_min_raw) * 0.1)
                pad_y_crop = int((y_max_raw - y_min_raw) * 0.1)
                
                x_min_crop = max(0, x_min_raw - pad_x_crop)
                x_max_crop = min(img_w, x_max_raw + pad_x_crop)
                y_min_crop = max(0, y_min_raw - pad_y_crop)
                y_max_crop = min(img_h, y_max_raw + pad_y_crop)
                
                face_crop = frame[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
                
                if face_crop.size > 0:
                    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(face_rgb)
                    input_tensor = val_transforms(pil_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        nn_score = model(input_tensor).item()
                else:
                    nn_score = 0.0

                # ── Fatigue Score from smoothed EAR ──
                # We revert to the mathematical MediaPipe EAR calculation for the live score.
                # The PyTorch model is currently only trained on 2 dummy images, so it will 
                # just output a constant score (e.g., 44) until trained on a real dataset.
                score = config.ear_to_fatigue_score(smoothed_ear)

                # ── Blink Detection (uses raw/instantaneous EAR) ──
                if avg_ear < EAR_BLINK_THRESHOLD:
                    blink_counter += 1

                    if eyes_closed_start is None:
                        eyes_closed_start = time.time()
                    eyes_closed_duration = time.time() - eyes_closed_start

                    # ── Drowsiness Alarm (sustained closure) ──
                    if eyes_closed_duration >= DROWSY_TIME_THRESHOLD:
                        if (time.time() - last_alarm_time) > ALARM_COOLDOWN:
                            threading.Thread(target=winsound.Beep, args=(ALARM_FREQUENCY, ALARM_DURATION_MS), daemon=True).start()
                            last_alarm_time = time.time()
                            print("WARNING: DROWSINESS ALARM! Wake up!")
                else:
                    if blink_counter >= BLINK_CONSEC_FRAMES:
                        session_blinks += 1
                    blink_counter = 0
                    eyes_closed_start = None
                    eyes_closed_duration = 0.0

                # ── Status Classification ──
                if score > 70:
                    color = (0, 0, 255)
                    status_text = "FATIGUED!"
                elif score > 40:
                    color = (0, 255, 255)
                    status_text = "Tired"
                else:
                    color = (0, 255, 0)
                    status_text = "Alert"

                # ── Draw face bounding box ──
                x_coords = [lm.x for lm in face_landmarks.landmark]
                y_coords = [lm.y for lm in face_landmarks.landmark]

                x_min = int(min(x_coords) * img_w)
                x_max = int(max(x_coords) * img_w)
                y_min = int(min(y_coords) * img_h)
                y_max = int(max(y_coords) * img_h)

                pad_x = int((x_max - x_min) * 0.05)
                pad_y = int((y_max - y_min) * 0.05)

                x_min = max(0, x_min - pad_x)
                x_max = min(img_w, x_max + pad_x)
                y_min = max(0, y_min - pad_y)
                y_max = min(img_h, y_max + pad_y)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                # ── Draw eye contours ──
                left_pts = [(int(pt.x * img_w), int(pt.y * img_h))
                            for pt in left_eye_pts]
                right_pts = [(int(pt.x * img_w), int(pt.y * img_h))
                             for pt in right_eye_pts]
                cv2.polylines(frame, [np.array(left_pts)], True, (0, 255, 255), 1)
                cv2.polylines(frame, [np.array(right_pts)], True, (0, 255, 255), 1)

                # ── Drowsiness warning flash ──
                if eyes_closed_duration >= DROWSY_TIME_THRESHOLD:
                    cv2.rectangle(frame, (0, 0), (img_w - 1, img_h - 1),
                                  (0, 0, 255), 8)
                    cv2.putText(frame, "!! WAKE UP !!",
                                (img_w // 2 - 120, img_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

            # ── Draw HUD Dashboard ──
            draw_dashboard(frame, score, status_text, color,
                           blink_counter, session_blinks,
                           eyes_closed_duration, fps, smoothed_ear)

            cv2.imshow("Eye Fatigue Estimator", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 50)
    print("   SESSION SUMMARY")
    print("=" * 50)
    print(f"  Total Blinks: {session_blinks}")
    print("=" * 50)


if __name__ == "__main__":
    run_inference()
