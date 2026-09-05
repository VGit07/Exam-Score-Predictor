# 🎓 Exam Score Predictor

A lightweight ML Web App built with **Scikit-Learn** and **Streamlit** that predicts a student's final exam score based on key academic and personal study habits.

---

## 📌 Project Overview
The goal of this project was to move away from overly complex pipelines and build an end-to-end ML project focused on clean regression, quick iteration, and deployment.

Starting from a baseline single-feature model ($R^2 = 0.48$), expanding to a multi-feature regression model boosted performance significantly to an **$R^2$ score of 0.61**.

---

## 📊 Model Performance & Metrics

The model evaluates predictions using standard continuous metrics, stored dynamically in `accuracy.json`:

| Metric | Score |
| :--- | :--- |
| **$R^2$ Score** | **0.61** |
| **MSE** | **139.09** |
| **RMSE** | **11.79** |
| **MAE** | **9.53** |
| **RMAE** | **3.09** |

---

## 🛠️ Project Structure

```text
Exam-Score-Predictor/
├── EDA.ipynb           # Jupyter Notebook for EDA of the Dataset
├── README.md           # About the Project
├── accuracy.json       # Exported model evaluation metrics
├── app.py              # Streamlit web UI script
├── data.csv            # Student performance dataset
├── model.pkl           # Saved Scikit-Learn model artifact
├── requirements.txt    # Python dependency manifest
└── train.py            # For Training, Testing and Saving of Model
```

---

## 🚀 Key Features of the App
* **Interactive Sliders & Inputs:** Adjust 5 input features (age,study hours, attendance, sleep, and internet access) in real time.
* **Live Prediction Engine:** Instant evaluation against `model.pkl`.
* **Dynamic Metric Display:** Directly loads performance stats from `accuracy.json` onto the sidebar UI.
* **Bounded Output:** Automatically caps outputs between 0 and 100 marks for realistic grading.

---

## 💻 Local Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/VGit07/Exam-Score-Predictor.git
   cd Exam-Score-Predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Training, Testing and Saving the Model**
   ```bash
   python train.py
   ```

4. **Run the Streamlit UI:**
   ```bash
   streamlit run app.py
   ```

---

## 🧰 Tech used in it
* **Language:** Python
* **ML Library:** Scikit-Learn
* **ML Model used :** Random Forest (Regression)
* **Data Processing:** Pandas, NumPy
* **Frontend / UI:** Streamlit
* **Model Serialization:** Joblib
