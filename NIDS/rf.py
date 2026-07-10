# ---------------- SYSTEM ----------------
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import warnings
warnings.filterwarnings("ignore")

# ---------------- IMPORTS ----------------
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# ---------------- COMMON ----------------
columns = [f"f{i}" for i in range(41)] + ["label", "difficulty"]

def clean(y):
    return y.astype(str).str.strip().str.replace(r'\.$', '', regex=True).str.lower()

attack_map = {
    'normal': 0,
    'neptune': 1, 'smurf': 1, 'back': 1, 'teardrop': 1, 'pod': 1, 'land': 1,
    'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'satan': 2,
    'ftp_write': 3, 'guess_passwd': 3, 'imap': 3, 'multihop': 3,
    'phf': 3, 'spy': 3, 'warezclient': 3, 'warezmaster': 3,
    'buffer_overflow': 4, 'loadmodule': 4, 'perl': 4, 'rootkit': 4
}

# ---------------- LOAD TRAIN ----------------
train_df = pd.read_csv("KDDTrain.csv", header=None)
train_df.columns = columns
train_df.drop("difficulty", axis=1, inplace=True)
train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)

X = train_df.drop("label", axis=1)
y = clean(train_df["label"]).map(attack_map)

mask = y.notnull()
X, y = X[mask], y[mask].astype(int)

# ---------------- ENCODING ----------------
categorical_cols = ["f1", "f2", "f3"]
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# ---------------- SPLIT ----------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------- SCALING ----------------
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# ---------------- BALANCING ----------------
print("Before balance:", Counter(y_train))

ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)

print("After balance:", Counter(y_train))

# ---------------- RANDOM FOREST MODEL ----------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=4
)

print("🚀 Training Random Forest...")
model.fit(X_train, y_train)

# ---------------- VALIDATION ----------------
y_val_pred = model.predict(X_val)

print("\n📊 VALIDATION")
print("Accuracy:", accuracy_score(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))

# ---------------- LOAD TEST ----------------
test_df = pd.read_csv("KDDTest.csv", header=None)
test_df.columns = columns
test_df.drop("difficulty", axis=1, inplace=True)
test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)

X_test = test_df.drop("label", axis=1)
y_test = clean(test_df["label"]).map(attack_map)

mask = y_test.notnull()
X_test, y_test = X_test[mask], y_test[mask].astype(int)

# ---------------- ENCODE TEST ----------------
for col, enc in encoders.items():
    X_test[col] = X_test[col].astype(str)
    X_test[col] = X_test[col].apply(
        lambda x: x if x in enc.classes_ else enc.classes_[0]
    )
    X_test[col] = enc.transform(X_test[col])

# ---------------- SCALE TEST ----------------
X_test = scaler.transform(X_test)

# ---------------- PREDICT TEST ----------------
y_test_pred = model.predict(X_test)

print("\n🔥 TEST")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
print("Distribution:", Counter(y_test_pred))

# ---------------- SAVE ----------------
joblib.dump(model, "rf_model.pkl")
joblib.dump(scaler, "rf_scaler.pkl")
joblib.dump(encoders, "rf_encoders.pkl")
joblib.dump(X.columns.tolist(), "rf_columns.pkl")

print("\n✅ RANDOM FOREST MODEL SAVED SUCCESSFULLY")