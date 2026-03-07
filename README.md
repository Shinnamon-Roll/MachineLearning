# Salmon vs Trout Classification Project

This project aims to classify images of Salmon and Trout using deep learning models. The core objective is to **compare three different models** to evaluate their performance in binary classification tasks.

## Project Structure

- **model-1/**: Contains the ImprovedDenseNet121 code (Research Based).
- **model-2/**: Contains the CustomMobileNetV2 code (Transfer Learning).
- **dashboard/**: Contains the Next.js web application for the user interface.
- **Image/**: Contains the dataset for training and testing.

## Dataset

The image dataset is located at:
`/Users/shinnamon/Documents/Project/MachineLearning/Image/`

The dataset is organized into:
- **Salmon!/**: Contains `Salmon Train` and `Salmon Test` folders.
- **Trout!/**: Contains `Trout Train` and `Trout Test` folders.

---

## 🔬 Model Comparison Plan

We will develop and compare three models:

1.  **Model 1: ImprovedDenseNet121 (Research Based)**
    - Status: ✅ Completed
    - Architecture: Transfer Learning with DenseNet121 (40% layers frozen).
    - Features: Binary classification (Salmon vs. Trout), Real-time training progress.

2.  **Model 2: CustomMobileNetV2**
    - Status: ✅ Implemented (Ready to Train)
    - Architecture: Transfer Learning with MobileNetV2 (40% layers frozen).
    - Features: Lightweight architecture, optimized for mobile/edge devices.

3.  **Model 3: (Pending)**
    - Status: 🚧 To be developed
    - Goal: Third architecture for comprehensive benchmarking.

---

## 🧠 Model 1: ImprovedDenseNet121

This model is adapted from a research paper and modified for binary classification.

### Training & Evaluation

To train Model 1:
```bash
cd model-1
python3 train.py
```

To evaluate Model 1:
```bash
cd model-1
python3 evaluate.py
```

---

## 📱 Model 2: CustomMobileNetV2

This model uses MobileNetV2, a lightweight architecture designed for mobile and embedded vision applications.

### Training & Evaluation

To train Model 2:
```bash
cd model-2
python3 train.py
```

To evaluate Model 2:
```bash
cd model-2
python3 evaluate.py
```

---

## 💻 Dashboard

The dashboard is a modern web application built with Next.js 14 to interact with the model and visualize results.

### Features
- **Real-time Inference**: Upload an image to classify it as Salmon or Trout.
- **Performance Visualization**: View accuracy, loss, and confusion matrix charts.
- **Dark Mode**: Toggle between light and dark themes for better visibility.
- **Model Comparison**: Compare metrics between Model 1 and Model 2 side-by-side.

### Installation & Running

1. Navigate to the `dashboard` directory:
   ```bash
   cd dashboard
   ```

2. Install dependencies (first time only):
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to:
   `http://localhost:3000`
