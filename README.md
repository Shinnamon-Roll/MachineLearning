# Salmon vs Trout Classification Project

This project aims to classify images of Salmon and Trout using deep learning models. The goal is to compare three different models, with the first model being based on the "ImprovedDenseNet121" research architecture.

## Project Structure

- **Machine/**: Contains the machine learning code (training, evaluation, data loading).
- **dashboard/**: Contains the Next.js web application for the user interface.
- **Image/**: Contains the dataset for training and testing.

## Dataset

The image dataset is located at:
`/Users/shinnamon/Documents/Project/MachineLearning/Image/`

The dataset is organized into:
- **Salmon!/**: Contains `Salmon Train` and `Salmon Test` folders.
- **Trout!/**: Contains `Trout Train` and `Trout Test` folders.

---

## Model 1: ImprovedDenseNet121 (Research Based)

This model is adapted from a research paper and modified for binary classification (Salmon vs. Trout). It uses transfer learning with a pre-trained DenseNet121 architecture, where 40% of the layers are frozen.

### Training

To train Model 1, navigate to the `Machine` directory and run the `train.py` script:

```bash
cd Machine
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
cd Machine
python3 evaluate.py
```

This will output:
- A confusion matrix.
- Precision, Recall, and F1-score for each class (Salmon and Trout).

---

## Model 2: (Pending)

*Status: To be developed.*

This section is reserved for the second model, which will be developed for comparison purposes.

---

## Model 3: (Pending)

*Status: To be developed.*

This section is reserved for the third model for comparison.

---

## Dashboard

The dashboard is a web application built with Next.js to interact with the model.

### Prerequisites
- Node.js and npm installed.

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
   `http://localhost:3001`

### Usage
- Upload an image of a Salmon or Trout.
- The dashboard will display the classification result (mock result currently, pending integration with the backend API).
