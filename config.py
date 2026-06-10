import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CSV_PATH = os.path.join(PROCESSED_DATA_DIR, "labels.csv")

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Hyperparameters
IMG_SIZE = (224, 224) # MobileNetV2 standard input size
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-4

# MediaPipe EAR mapping to Fatigue Score (0-100)
# EAR typically ranges from ~0.20 (eyes closed) to ~0.35 (eyes wide open)
EAR_MAX = 0.35 # Fatigue 0
EAR_MIN = 0.20 # Fatigue 100

def ear_to_fatigue_score(ear):
    """
    Linearly maps EAR to a fatigue score from 0 to 100.
    A lower EAR means the eyes are more closed (higher fatigue).
    """
    if ear >= EAR_MAX:
        return 0.0
    if ear <= EAR_MIN:
        return 100.0
    
    # Linear interpolation
    # (EAR_MAX - ear) / (EAR_MAX - EAR_MIN) maps to 0.0 - 1.0
    score = ((EAR_MAX - ear) / (EAR_MAX - EAR_MIN)) * 100.0
    return round(score, 2)
