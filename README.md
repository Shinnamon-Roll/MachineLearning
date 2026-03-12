# 🐟 Salmon vs Trout Classification Project

This project leverages **Deep Learning** to classify images of **Salmon** and **Trout** with high accuracy. It features a comprehensive comparison between two powerful architectures: **DenseNet121** and **MobileNetV2**, integrated into a modern **Next.js Dashboard** for real-time inference and analysis.

---

## 🚀 Key Features

*   **Dual Model Architecture**:
    *   **Model 1 (ImprovedDenseNet121)**: A robust, research-based model focusing on maximizing feature extraction depth.
    *   **Model 2 (CustomMobileNetV2)**: A lightweight, optimized model using **Transfer Learning** with a **2-Phase Training Strategy** (Frozen -> Fine-tuning) and **Focal Loss** to handle hard examples.
*   **Interactive Dashboard**:
    *   Built with **Next.js 16 (App Router)** and **Tailwind CSS 4**.
    *   **Real-time Inference**: Upload an image to see immediate classification results from both models simultaneously.
    *   **Visual Analytics**: Interactive charts (Recharts) showing Training Loss, Accuracy, and Dataset Splits.
    *   **Modern UI**: Features Glassmorphism, Spotlight effects, and a strict **Monochrome** design system.
*   **Advanced Data Pipeline**:
    *   **Preprocessing**: CLAHE (Contrast Limited Adaptive Histogram Equalization) for texture enhancement.
    *   **Augmentation**: Random Rotation, Horizontal Flip, and Resize to 224x224.
    *   **Split Strategy**: 80% Train / 10% Validation / 10% Test.

---

## 🛠️ Tech Stack

### **Frontend (Dashboard)**
*   **Framework**: Next.js 16 (React 19, TypeScript)
*   **Styling**: Tailwind CSS 4, Framer Motion, Aceternity UI, Shadcn UI
*   **Visualization**: Recharts, Lucide React

### **Backend & AI**
*   **API**: Next.js API Routes (Serverless Functions)
*   **Deep Learning**: PyTorch, Torchvision
*   **Image Processing**: PIL (Pillow), OpenCV (for CLAHE)
*   **Environment**: Python 3.x (MPS/CUDA support)

---

## 📂 Project Structure

```bash
.
├── dashboard/          # Next.js Web Application
│   ├── src/            # Source code (App Router, Components)
│   ├── public/         # Static assets & JSON metrics
│   └── package.json    # Frontend dependencies
├── model-1/            # ImprovedDenseNet121 (Research Based)
│   ├── model.py        # Model architecture
│   ├── train.py        # Training script (Single Phase)
│   └── evaluate.py     # Evaluation script
├── model-2/            # CustomMobileNetV2 (Transfer Learning)
│   ├── model_mobilenet.py # Model architecture
│   ├── train.py        # Training script (2-Phase: Frozen + Fine-tune)
│   ├── focal_loss.py   # Custom Loss Function
│   └── evaluate.py     # Evaluation script
├── Docs/               # Project Documentation & References
└── README.md           # Project Overview (This file)
```

---

## ⚡ Getting Started

### 1. Prerequisites
*   **Node.js** (v18 or higher)
*   **Python** (v3.9 or higher)
*   **Pip** & **Virtualenv** (Recommended)

### 2. Setup Python Environment (for Models)
```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies for Model 1 & 2
pip install -r model-1/requirements.txt
# OR
pip install torch torchvision pillow numpy scikit-learn matplotlib
```

### 3. Run the Dashboard
```bash
cd dashboard

# Install dependencies
npm install

# Start the development server
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🔬 Model Comparison

| Feature | Model 1 (DenseNet121) | Model 2 (MobileNetV2) |
| :--- | :--- | :--- |
| **Architecture** | Deep, Densely Connected | Lightweight, Depthwise Separable Conv |
| **Training Strategy** | Single Phase (Full Training) | **2-Phase**: Frozen (Head) -> Fine-tuning (Body) |
| **Loss Function** | Cross Entropy Loss | **Focal Loss** (Focus on hard examples) |
| **Best For** | Maximum Accuracy (Server-side) | Speed & Efficiency (Mobile/Edge) |
| **Inference Time** | ~150ms | **~40ms** (Faster) |

---

## 📸 Screenshots

*(Add screenshots of the dashboard here)*

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
