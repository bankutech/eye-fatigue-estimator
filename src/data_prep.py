import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# pyre-ignore[21]
import config

mp_face_mesh = mp.solutions.face_mesh

# Eye landmark indices for MediaPipe Face Mesh
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/geometry_pipeline_metadata_landmarks.csv
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def compute_ear(eye_landmarks, img_w, img_h):
    """
    Computes Eye Aspect Ratio (EAR) given eye landmarks.
    """
    # Convert normalized coordinates to pixel coordinates
    pts = [(int(pt.x * img_w), int(pt.y * img_h)) for pt in eye_landmarks]
    
    # EAR formula: (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    # Note: MediaPipe returns landmarks in a specific order, we adjust to standard EAR
    # p1=Left, p4=Right, p2/p3=Top, p5/p6=Bottom
    # For LEFT_EYE array: 0:Right, 1:Top-Right, 2:Top-Left, 3:Left, 4:Bottom-Left, 5:Bottom-Right
    # To match EAR standard: p1=pt[0], p2=pt[1], p3=pt[2], p4=pt[3], p5=pt[4], p6=pt[5]
    
    vertical_1 = get_distance(pts[1], pts[5])
    vertical_2 = get_distance(pts[2], pts[4])
    horizontal = get_distance(pts[0], pts[3])
    
    if horizontal == 0:
        return 0.0
        
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def process_images():
    print(f"Reading raw images from {config.RAW_DATA_DIR}")
    
    image_files = [f for f in os.listdir(config.RAW_DATA_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in raw data directory. Please add some images.")
        return

    records = []
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        for idx, img_name in enumerate(tqdm(image_files)):
            img_path = os.path.join(config.RAW_DATA_DIR, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            img_h, img_w, _ = image.shape
            # MediaPipe expects RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)
            
            # pyre-ignore[16]
            if not results.multi_face_landmarks:
                continue
                
            # pyre-ignore[16]
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract eye landmarks
            left_eye_pts = [face_landmarks.landmark[i] for i in LEFT_EYE]
            right_eye_pts = [face_landmarks.landmark[i] for i in RIGHT_EYE]
            
            left_ear = compute_ear(left_eye_pts, img_w, img_h)
            right_ear = compute_ear(right_eye_pts, img_w, img_h)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Convert EAR to fatigue score (0-100)
            fatigue_score = config.ear_to_fatigue_score(avg_ear)
            
            # Crop Face
            # Find bounding box of face landmarks
            x_min = min([lm.x for lm in face_landmarks.landmark])
            x_max = max([lm.x for lm in face_landmarks.landmark])
            y_min = min([lm.y for lm in face_landmarks.landmark])
            y_max = max([lm.y for lm in face_landmarks.landmark])
            
            # Add some padding
            pad_x = (x_max - x_min) * 0.1
            pad_y = (y_max - y_min) * 0.1
            
            x_min = max(0, x_min - pad_x)
            x_max = min(1, x_max + pad_x)
            y_min = max(0, y_min - pad_y)
            y_max = min(1, y_max + pad_y)
            
            px_min, px_max = int(x_min * img_w), int(x_max * img_w)
            py_min, py_max = int(y_min * img_h), int(y_max * img_h)
            
            face_crop = image[py_min:py_max, px_min:px_max]
            
            if face_crop.size == 0:
                continue
                
            # Resize face to target model input size (e.g., 224x224)
            face_resized = cv2.resize(face_crop, config.IMG_SIZE)
            
            # Save processed image
            out_filename = f"processed_{idx:05d}.jpg"
            out_path = os.path.join(config.PROCESSED_DATA_DIR, out_filename)
            cv2.imwrite(out_path, face_resized)
            
            records.append({
                "filename": out_filename,
                "fatigue_score": fatigue_score
            })
            
    # Save records to CSV
    df = pd.DataFrame(records)
    df.to_csv(config.CSV_PATH, index=False)
    print(f"Processed {len(records)} images. Labels saved to {config.CSV_PATH}")

if __name__ == "__main__":
    process_images()
