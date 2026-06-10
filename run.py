import os
import sys
import argparse

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# pyre-ignore[21]
from src.data_prep import process_images
# pyre-ignore[21]
from src.train import train_model
# pyre-ignore[21]
from src.evaluate import evaluate_model
# pyre-ignore[21]
from src.inference import run_inference

def main():
    parser = argparse.ArgumentParser(description="Eye Fatigue Estimator Pipeline Runner")
    parser.add_argument("--mode", type=str, default="all", 
                        choices=["all", "train-only", "live-only"],
                        help="all: Run data prep, train, eval, and live inference. "
                             "train-only: Prep, train, and evaluate without webcam. "
                             "live-only: Run live webcam inference directly.")
    
    args = parser.parse_args()
    
    if args.mode == "live-only":
        print("\n=== Running Live Webcam Inference ===")
        run_inference()
        return

    # Run full training pipeline
    print("=== Step 1: Processing Images ===")
    process_images()
    
    print("\n=== Step 2: Training Model ===")
    train_model()
    
    print("\n=== Step 3: Evaluating Model ===")
    evaluate_model()
    
    if args.mode == "all":
        print("\n=== Step 4: Running Live Webcam Inference ===")
        run_inference()

if __name__ == "__main__":
    main()
