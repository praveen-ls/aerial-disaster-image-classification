
# Enhancing Emergency Response with Aerial Image Classification

## 🔍 Project Summary 

This project applies **deep learning and transfer learning** to classify aerial disaster images for emergency response. Using the **AIDER dataset**, multiple CNN architectures and feature-based classifiers were evaluated to identify disaster types such as **fire, flood, collapsed buildings, traffic accidents, and normal scenes**. The work focuses on **model performance, class imbalance handling, and interpretability**, making it relevant for real-world AI systems deployed in high-stakes environments.

---

## 📌 Problem Statement

Manual inspection of aerial images during disasters is slow and error-prone. This project aims to **automate disaster recognition from aerial imagery**, enabling faster situational awareness and improved emergency response planning.

---

## 📂 Dataset

* **Dataset:** AIDER (Aerial Image Database for Emergency Response)
* **Classes:**

  * Fire
  * Flood
  * Collapsed Buildings
  * Traffic Accidents
  * Normal
* **Key Challenge:** Severe class imbalance (Normal class dominates)

---

## 🧹 Data Preprocessing & Augmentation

* Image resizing to **256 × 256**
* Class-wise **80/20 train–test split**
* Targeted data augmentation for disaster classes:

  * Rotation
  * Translation
  * Zoom
  * Shear
  * Horizontal/vertical flipping
* Balanced each disaster class to ~3200 samples

---

## 🧠 Models & Methodology

### 1️⃣ Baseline CNN

* Custom CNN built from scratch
* Adam optimizer, categorical cross-entropy loss
* Used as a baseline for comparison

### 2️⃣ Transfer Learning Models

* **ResNet-50**
* **EfficientNet-B0**
* **ConvNeXt-Base**
* ImageNet pretrained weights
* Fine-tuned final layers for disaster classification

### 3️⃣ Feature-Based Machine Learning

* Extracted **2048-dimensional features** from ResNet-50
* Applied classical ML classifiers:

  * Logistic Regression
  * Linear SVM
  * KNN
  * Random Forest
  * XGBoost

---

## 🔍 Model Interpretability

* Generated **gradient-based saliency maps**
* Visualized spatial regions influencing predictions
* Helped diagnose:

  * Background bias
  * Class confusion
  * Misclassification patterns

---

## ⚙️ Hyperparameter Tuning

Grid search over:

* Learning rate: `0.001`, `0.0005`
* Batch size: `32`, `64`
* Dropout rate: `0.3`, `0.5`

Evaluation emphasized **F1-score** due to class imbalance.

---

## 📊 Results

| Model                           | Accuracy |
| ------------------------------- | -------- |
| Tuned CNN                       | ~81%     |
| ResNet-50 (fine-tuned)          | ~96%     |
| ResNet-50 + Logistic Regression | ~96%     |
| ResNet-50 + SVM                 | ~95.9%   |

* Transfer learning significantly outperformed training from scratch
* Feature extraction + ML provided strong performance with lower training cost
* ROC-AUC > 0.90 for most classes

---

## 📈 Visual Results

* Confusion matrices

  __Resnet Confusion Matrix__
  <figure>
  <img width="784" height="712" alt="description" src="https://github.com/user-attachments/assets/4922b704-6e68-41d6-8593-97e642c28f41" />
  </figure>


  __CNN confusion matrix__
  <figure>
  <img width="784" height="712" alt="description" src="https://github.com/user-attachments/assets/16912b6a-9293-4f89-8407-1cbcc69bd18f" />
</figure>

* ROC curves
  
  __Resnet ROC curve__
  
  <img width="852" height="704" alt="image" src="https://github.com/user-attachments/assets/f306135f-5a62-4afe-8f2a-c0624317d4bb" />

  
  __CNN ROC Curve__
  
  <img width="848" height="701" alt="image" src="https://github.com/user-attachments/assets/b4bebf7f-cea8-46ea-8fe1-749803066c3e" />

  
     


* Saliency map visualizations
  
 <img width="547" height="623" alt="image" src="https://github.com/user-attachments/assets/117cce63-e404-4b69-9f26-aa6ecd4f4845" />

 <img width="548" height="622" alt="image" src="https://github.com/user-attachments/assets/41f4a674-c7df-47be-be67-1b7a75825326" />

 <img width="544" height="622" alt="image" src="https://github.com/user-attachments/assets/9549a728-f05b-481e-a4e4-551f67e2c3ff" />


---

## 🛠️ Technologies Used

* **Languages:** Python
* **Deep Learning:** TensorFlow, PyTorch
* **ML:** Scikit-learn
* **Data:** NumPy, Pandas
* **Visualization:** Matplotlib
* **Concepts:** CNNs, Transfer Learning, Feature Extraction, Model Interpretability

---

## ▶️ How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/aerial-disaster-classification.git
cd aerial-disaster-classification
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train models

```bash
python train_cnn.py
python train_resnet.py
```

### 4️⃣ Feature-based classification

```bash
python extract_features.py
python train_ml_models.py
```

### 5️⃣ Generate saliency maps

```bash
python saliency_maps.py
```

---

## 💡 Key Learnings

* Transfer learning is highly effective for limited, imbalanced datasets
* Feature-based ML offers a strong accuracy-vs-cost tradeoff
* Model interpretability is critical for safety-critical AI applications
* Increasing model complexity does not always improve performance

---

## 👥 Authors

* **Praveen Lakshman** ([@praveen-ls](https://github.com/praveen-ls))
* Achuth Reddy Bangaru ([@achuthreddy-16](https://github.com/achuthreddy-16))
* Aneesh Gundeti ([@Aneeshg02](https://github.com/Aneeshg02))
* Manish Krishna Nalluru

University of Alabama at Birmingham

---

## 📚 References

* Kyrkou & Theocharides, *EmergencyNet*, IEEE JSTARS, 2020
* Kyrkou & Theocharides, *Deep Learning for UAV Emergency Response*, CVPR Workshops, 2019

---
