# ---------------- SYSTEM ----------------
import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import warnings
warnings.filterwarnings("ignore")

# ---------------- IMPORTS ----------------
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from collections import Counter

# ---------------- CONFIG ----------------
columns = [f"f{i}" for i in range(41)] + ["label", "difficulty"]

attack_map = {
    'normal': 0,
    'neptune': 1, 'smurf': 1, 'back': 1, 'teardrop': 1, 'pod': 1, 'land': 1,
    'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'satan': 2,
    'ftp_write': 3, 'guess_passwd': 3, 'imap': 3, 'multihop': 3,
    'phf': 3, 'spy': 3, 'warezclient': 3, 'warezmaster': 3,
    'buffer_overflow': 4, 'loadmodule': 4, 'perl': 4, 'rootkit': 4
}

def clean(y):
    return y.astype(str).str.strip().str.replace(r'\.$', '', regex=True).str.lower()

# ---------------- LOAD ----------------
df = pd.read_csv("KDDTrain.csv", header=None)
df.columns = columns
df.drop("difficulty", axis=1, inplace=True)

X = df.drop("label", axis=1)
y = clean(df["label"]).map(attack_map)

mask = y.notnull()
X, y = X[mask], y[mask].astype(int)

# ---------------- ENCODING ----------------
encoders = {}
for col in ["f1", "f2", "f3"]:
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

# ---------------- SMOTE ----------------
print("Before SMOTE:", Counter(y_train))

smote = SMOTE(
    sampling_strategy={3: 3000, 4: 1500},
    k_neighbors=3,
    random_state=42
)

X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:", Counter(y_train))

# ---------------- MAIN MODEL ----------------
model = XGBClassifier(
    n_estimators=650,
    max_depth=8,
    learning_rate=0.04,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.15,
    min_child_weight=1,
    reg_alpha=0.4,
    reg_lambda=3,
    objective="multi:softprob",
    num_class=5,
    eval_metric="mlogloss",
    tree_method="hist",
    n_jobs=1
)

print("🚀 Training Main Model...")
model.fit(X_train, y_train)

# ---------------- RARE ATTACK MODELS ----------------

# R2L
y_r2l = (y_train == 3).astype(int)
model_r2l = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=(len(y_r2l) - sum(y_r2l)) / sum(y_r2l),
    n_jobs=1
)
model_r2l.fit(X_train, y_r2l)

# U2R
y_u2r = (y_train == 4).astype(int)
model_u2r = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=(len(y_u2r) - sum(y_u2r)) / sum(y_u2r),
    n_jobs=1
)
model_u2r.fit(X_train, y_u2r)

# ---------------- FINAL PREDICTION ----------------
def predict_final(X):

    main_probs = model.predict_proba(X)
    main_preds = np.argmax(main_probs, axis=1)

    r2l_probs = model_r2l.predict_proba(X)
    u2r_probs = model_u2r.predict_proba(X)

    preds = []

    for i in range(len(X)):

        if u2r_probs[i][1] > 0.15:
            preds.append(4)
            continue

        if r2l_probs[i][1] > 0.25:
            preds.append(3)
            continue

        preds.append(main_preds[i])

    return np.array(preds)

# ---------------- EVALUATION FUNCTION ----------------
def evaluate_model(y_true, y_pred, title="RESULT"):

    print(f"\n📊 {title}")

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print("Accuracy :", acc)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)

    print("\n Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\n" \
    " Classification Report:")
    print(classification_report(y_true, y_pred))

# ---------------- VALIDATION ----------------
y_val_pred = predict_final(X_val)
evaluate_model(y_val, y_val_pred, "VALIDATION")

# ---------------- TEST ----------------
test_df = pd.read_csv("KDDTest.csv", header=None)
test_df.columns = columns
test_df.drop("difficulty", axis=1, inplace=True)

X_test = test_df.drop("label", axis=1)
y_test = clean(test_df["label"]).map(attack_map)

mask = y_test.notnull()
X_test, y_test = X_test[mask], y_test[mask].astype(int)

for col, enc in encoders.items():
    X_test[col] = X_test[col].astype(str)
    X_test[col] = X_test[col].apply(lambda x: x if x in enc.classes_ else enc.classes_[0])
    X_test[col] = enc.transform(X_test[col])

X_test = scaler.transform(X_test)

y_test_pred = predict_final(X_test)

evaluate_model(y_test, y_test_pred, "TEST")

print("\n Prediction Distribution:")
print(Counter(y_test_pred))

#--------------- SAVE ----------------
joblib.dump(model, "model.pkl")
joblib.dump(model_r2l, "r2l_model.pkl")
joblib.dump(model_u2r, "u2r_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("\nFINAL NIDS MODEL saved successfully!")