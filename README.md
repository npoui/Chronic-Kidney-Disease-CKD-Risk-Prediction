# Chronic Kidney Disease (CKD) Risk Prediction

This project predicts **Chronic Kidney Disease (CKD) risk** based on patient clinical data and nephrotoxic drug information using Machine Learning models.

The dataset is sourced from Kaggle:
**CKD Nephrotoxic Drug Dataset**

---

## 📊 Dataset

- Source: Kaggle
- File used:
  - `CKD_NephrotoxicDrug_Dataset.csv`
- Target variable:
  - `ckd_risk_label`
    - 0 → No CKD
    - 1 → CKD

Dataset is **not included** in the repository due to size constraints.

---

## 🧠 Models Used

- **Random Forest Classifier**
- **K-Nearest Neighbors (KNN)**

---

## ⚙️ Methodology

1. Load and inspect dataset
2. Handle missing values and duplicates
3. Exploratory data analysis (EDA)
4. One-hot encoding of categorical variables
5. Feature scaling using StandardScaler
6. Train-test split (80–20)
7. Model training
8. Performance evaluation using accuracy score
9. CKD prediction for new patient input
