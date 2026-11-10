# TruthLens — Phase 1 Implementation Starter

This starter implements the first milestone of the TruthLens project:
image -> preprocessing -> CNN/ResNet -> REAL/FAKE prediction.

## Project structure

TruthLens/
├── dataset/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
├── models/
├── results/
├── src/
│   ├── train.py
│   ├── predict.py
│   └── utils.py
├── requirements.txt
└── .gitignore

## 1. Create environment (Windows)

python -m venv venv
venv\Scripts\activate

## 2. Install dependencies

pip install -r requirements.txt

## 3. Add data

Put labelled images into:

dataset/train/real
dataset/train/fake
dataset/val/real
dataset/val/fake
dataset/test/real
dataset/test/fake

Keep the same class folder names: `real` and `fake`.

## 4. Train

python src/train.py

The trained model is saved as:
models/truthlens_resnet18.pth

## 5. Predict a new image

python src/predict.py path/to/image.jpg

## First milestone

The expected output is similar to:

Prediction: FAKE
Confidence: 87.42%

This is only the first module. After it works, the next milestones are:
1. video frame extraction
2. audio analysis
3. lip-sync analysis
4. multimodal fusion
5. FastAPI backend
6. React/Next.js frontend
7. database/storage
8. explainability and evaluation
