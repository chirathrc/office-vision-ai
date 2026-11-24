# OfficeVision AI — Real-Time Workplace Activity & Custom Trainable Vision App

A browser-based real-time camera & photo AI application built with **React, TypeScript, Vite, Tailwind CSS, and TensorFlow.js** that identifies office activities (`Working at PC`, `Using Phone`, `Reading a Book`, `On Break / Drinking`, `Idle / Away`) and lets you train custom classes live.

## Features
- **Pre-Trained Custom AI Model Included**: Ships with `public/pretrained-office-model.json` (304 mirror-augmented MobileNet v2 embeddings, 73.0% LOO cross-validation accuracy) + locally hosted MobileNet v2 weights in `public/models/mobilenet_v2/` for sub-second startup.
- **Live Camera & Photo Upload**: Run real-time detection on your webcam or drag-and-drop / upload any office photo.
- **In-App Custom Model Trainer**: Add any custom class, capture single frames, `Burst +15` frames, or upload training photos (`Photos`), auto-save to browser storage, and export/import trained `.json` weights.
- **Optional Gemini Vision Deep Analysis**: Paste a Gemini API key to run detailed natural-language scene analysis on any frame.
- **Offline Dataset & YOLOv8 Training Scripts**: Includes `training/download_and_train.mjs` (TensorFlow.js MobileNet v2 trainer) and `training/train_custom_yolo.py` (Ultralytics YOLOv8).

## Quick Start
```bash
npm install
npm run dev
```
Open `http://localhost:5173`.
