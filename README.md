# Fashion-MNIST Classification Project

## 📌 Description
This project implements a deep learning model to classify **Fashion-MNIST images** into 10 categories.
It follows the methodology from the research paper **"Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms"** by Han Xiao, Kashif Rasul, and Roland Vollgraf (Zalando Research).

## 📂 Dataset
**Fashion-MNIST** is a dataset consisting of **70,000 grayscale images (28×28 pixels)** across 10 categories.
- **Train Set**: 60,000 images
- **Test Set**: 10,000 images
- **Classes**:
  ```
  0 - T-shirt/top
  1 - Trouser
  2 - Pullover
  3 - Dress
  4 - Coat
  5 - Sandal
  6 - Shirt
  7 - Sneaker
  8 - Bag
  9 - Ankle boot
  ```
- **Download**: [Fashion-MNIST Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

## ⚡ Model Implementation
This project implements a **fine-tuned deep learning model** using pre-trained architectures for Fashion-MNIST classification.

### **✅ Features**
- **Pre-trained models**: MobileNetV2 / VGG16
- **Fine-tuning**: Freezing initial layers & optimizing top layers
- **Data Augmentation**: Random rotations, flips, zoom
- **Optimizer**: Adam / RMSprop
- **Learning Rate Scheduling**: `ReduceLROnPlateau`
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score, Confusion Matrix

## 🚀 How to Run
### **1️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2️⃣ Train the Model**
```bash
python train.py
```

### **3️⃣ Evaluate the Model**
```bash
python evaluate.py
```

## 📊 Results & Performance
| **Metric**     | **Research Paper (SVM, MLP, CNN)** | **Our Model (Fine-tuned CNN)** |
|---------------|---------------------------------|---------------------------|
| **Model Used** | SVM, MLP, CNN                   | MobileNetV2 / VGG16       |
| **Accuracy**   | **89.7%** (SVM)                 | **XX%**                   |
| **Loss**       | Not mentioned                   | **XX**                     |
| **Optimizer**  | SGD, SVM, Adam                  | Adam / RMSprop            |
| **Dataset Size** | 70,000 images                | 70,000 images             |

## 📸 Performance Visuals
1️⃣ **Training & Validation Accuracy/Loss Plots**
2️⃣ **Confusion Matrix Analysis**
3️⃣ **Feature Map Visualizations**

## 🛠️ Possible Improvements
✅ **Memory Optimization**: Use `tf.data.Dataset` with `.batch()` and `.cache()` to reduce RAM usage.
✅ **Reduce Overfitting**: Apply Dropout, L2 regularization.
✅ **Hyperparameter Tuning**: Experiment with different optimizers & learning rates.
✅ **Edge Deployment**: Apply TensorFlow Lite quantization for mobile devices.

---

📌 **Author**: Suraj Didwagh 
