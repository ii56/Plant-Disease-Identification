# Plant Disease Identification

This project implements a plant disease identification system using classical image
features and a Random Forest classifier. It includes model training, evaluation,
and real-world image inference with confidence scoring.

## Project Structure

- `main.ipynb`  
  Training, validation, testing, and model saving

- `prepare_dataset.py`  
  Dataset preparation and splitting

- `plant_disease_inference.py`  
  Standalone script for real-world image prediction

- `rf_plant_disease_balanced.pkl`  
  Trained Random Forest model

- `feature_scaler.pkl`, `label_encoder.pkl`  
  Saved preprocessing artifacts

## Dataset

The model was trained using the PlantVillage dataset:  
https://www.kaggle.com/datasets/emmarex/plantdisease

## How to Run Inference

```bash
python plant_disease_inference.py

```
