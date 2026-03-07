# Salmon vs Trout Classification Project

This project aims to classify images of Salmon and Trout using deep learning models. The core objective is to **compare three different models** to evaluate their performance in binary classification tasks.

## Project Structure

- **machine/**: Contains the machine learning code (training, evaluation, data loading).
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

2.  **Model 2: (Pending)**
    - Status: 🚧 To be developed
    - Goal: Alternative architecture for comparison.

3.  **Model 3: (Pending)**
    - Status: 🚧 To be developed
    - Goal: Third architecture for comprehensive benchmarking.

---

## 🧠 Model 1: ImprovedDenseNet121

This model is adapted from a research paper and modified for binary classification.

### Training

To train Model 1, navigate to the `machine` directory and run the `train.py` script:

```bash
cd machine
python3 train.py
```

During training, the script will display:
- Real-time progress with the name of the image being processed.
- Loss and Accuracy for both training and validation phases.
- A success message upon completion.

The best model weights will be saved as `best_model.pth`.

### Testing / Evaluation

To evaluate the trained model on the test dataset:

```bash
cd machine
python3 evaluate.py
```

This will output:
- A confusion matrix.
- Precision, Recall, and F1-score for each class.
- Results will also be saved to the dashboard for visualization.

---

## 💻 Dashboard

The dashboard is a modern web application built with Next.js 14 to interact with the model and visualize results.

### Features
- **Real-time Inference**: Upload an image to classify it as Salmon or Trout.
- **Performance Visualization**: View accuracy, loss, and confusion matrix charts.
- **Dark Mode**: Toggle between light and dark themes for better visibility.

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
   `http://localhost:3000` (or the port shown in your terminal)
