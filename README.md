# 🧠 Student Mental Stress Prediction

This project is a machine learning-based web application that predicts the stress level of students based on various academic, financial, and psychological factors using XGBoost. It includes a Flask API backend and a user-friendly HTML frontend.

## 🚀 Features

- Predicts stress levels: Low, Moderate, High
- Trained using synthetic labeled data
- Clean web interface for input
- Real-time predictions via Flask API
- Scalable and modular project structure

## 📊 Input Features

- Academic Workload (1–5)
- Sleep Quality (1–5)
- Financial Strain (1–5)
- Social Support (1–5)
- Anxiety Level (1–5)

## 🛠️ Tech Stack

- Python 3
- Flask & Flask-CORS
- XGBoost
- Scikit-learn
- Pandas & NumPy
- HTML/CSS for frontend

## 📁 Project Structure

```
Student-Mental-Stress-Prediction/
├── app/
│   ├── Flask_api.py          # Flask server handling prediction requests
│   ├── modell.py             # Model training and saving script
│   ├── xgb_stress_model.pkl  # Trained XGBoost model
│   ├── stress_scaler.pkl     # StandardScaler used in preprocessing
├── templates/
│   └── frontend.html         # Web interface for inputs
├── static/                   # (Optional) CSS/JS files
├── requirements.txt
├── .gitignore
├── README.md
```

## ⚙️ How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/VishalRagam/Student-Mental-Stress-Prediction.git
cd Student-Mental-Stress-Prediction
```

### 2. Set up virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Flask API

```bash
cd app
python Flask_api.py
```

### 4. Open the Frontend

Open `templates/frontend.html` in your browser.  
Fill in the form and click **Predict Stress Level** to see the result.

## 📌 Note

- You can retrain the model using `modell.py` if needed.
- Make sure both `xgb_stress_model.pkl` and `stress_scaler.pkl` are present.

## ✍️ Author

**Ragam Vishal**  
Project under personal ML practice and academic exploration.
