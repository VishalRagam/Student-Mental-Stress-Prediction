import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib

# Generate synthetic data
np.random.seed(42)
num_samples = 500

academic_workload = np.random.randint(1, 6, num_samples)
sleep_quality = np.random.randint(1, 6, num_samples)
financial_strain = np.random.randint(1, 6, num_samples)
social_support = np.random.randint(1, 6, num_samples)
anxiety_level = np.random.randint(1, 6, num_samples)

stress_score = (academic_workload * 1.2 +
                (6 - sleep_quality) * 1.1 +
                financial_strain * 1.3 +
                (6 - social_support) * 1.0 +
                anxiety_level * 1.5 +
                np.random.normal(0, 2, num_samples))

# Balanced class labels using quantiles
stress_level = pd.qcut(stress_score, q=3, labels=[0, 1, 2]).astype(int)

df = pd.DataFrame({
    'AcademicWorkload': academic_workload,
    'SleepQuality': sleep_quality,
    'FinancialStrain': financial_strain,
    'SocialSupport': social_support,
    'AnxietyLevel': anxiety_level,
    'StressLevel': stress_level
})

X = df.drop('StressLevel', axis=1)
y = df['StressLevel']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

xgb_clf = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    eval_metric='mlogloss',
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

xgb_clf.fit(X_train, y_train)

y_pred = xgb_clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))

joblib.dump(xgb_clf, 'xgb_stress_model.pkl')
joblib.dump(scaler, 'stress_scaler.pkl')
