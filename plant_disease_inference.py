import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
from skimage.feature import graycomatrix, graycoprops

# Paths to saved model artifacts
# --------------------------------------------------
BASE_DIR = Path(r"C:\Users\HP\OneDrive\Apps\Documents\Codes\ML\Plant Disease")

MODEL_PATH  = BASE_DIR / "rf_plant_disease_balanced.pkl"
SCALER_PATH = BASE_DIR / "feature_scaler.pkl"
ENC_PATH    = BASE_DIR / "label_encoder.pkl"

# Load model, scaler, encoder
# --------------------------------------------------
rf_model = joblib.load(MODEL_PATH)
scaler   = joblib.load(SCALER_PATH)
encoder  = joblib.load(ENC_PATH)

print("Model loaded successfully")
print("Classes:", encoder.classes_)

# --------------------------------------------------
# Image loading
# --------------------------------------------------
def load_image(path, size=(128, 128)):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img

# Feature extraction
# --------------------------------------------------
def extract_color_features(img):
    feats = []

    for i in range(3):
        feats.append(np.mean(img[:, :, i]))
        feats.append(np.std(img[:, :, i]))

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    for i in range(3):
        feats.append(np.mean(hsv[:, :, i]))
        feats.append(np.std(hsv[:, :, i]))

    return feats


def extract_texture_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=256,
        symmetric=True,
        normed=True
    )

    return [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ]


def extract_features(image_path):
    img = load_image(image_path)
    return extract_color_features(img) + extract_texture_features(img)

# Prediction with confidence
# --------------------------------------------------
def predict_image_with_confidence(image_path):
    features = extract_features(image_path)
    features = np.array(features).reshape(1, -1)

    features_scaled = scaler.transform(features)

    pred_encoded = rf_model.predict(features_scaled)[0]
    pred_label = encoder.inverse_transform([pred_encoded])[0]

    probs = rf_model.predict_proba(features_scaled)[0]
    confidence = probs[pred_encoded]

    return pred_label, confidence, probs

# File Explorer
# --------------------------------------------------
root = tk.Tk()
root.withdraw()
root.wm_attributes("-topmost", 1)

def select_image_file():
    file_path = filedialog.askopenfilename(
        title="Select a leaf image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )

    if not file_path:
        raise RuntimeError("No file selected")

    return Path(file_path)

# Main execution
# --------------------------------------------------
def main():
    test_image = select_image_file()
    print("Selected image:", test_image)

    label, confidence, probs = predict_image_with_confidence(test_image)

    print("\nPredicted disease:", label)
    print(f"Confidence: {confidence * 100:.2f}%")

    print("\nTop 5 predictions:")
    top_indices = np.argsort(probs)[::-1][:5]
    for idx in top_indices:
        class_name = encoder.inverse_transform([idx])[0]
        print(f"{class_name:45s} {probs[idx] * 100:.2f}%")

    img = cv2.imread(str(test_image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"{label} ({confidence * 100:.1f}%)")
    plt.show()


if __name__ == "__main__":
    main()
