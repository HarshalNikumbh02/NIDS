import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load NSL-KDD training dataset
data = pd.read_csv("KDDTrain.csv", header=None)

# Last column is attack label
X = data.iloc[:, :-2]
y = data.iloc[:, -2]

# Convert attack types to categories
attack_map = {
    "normal":0,
    "neptune":1,
    "smurf":1,
    "teardrop":1,
    "satan":2,
    "ipsweep":2,
    "portsweep":2,
    "guess_passwd":3,
    "ftp_write":3,
    "buffer_overflow":4,
    "rootkit":4
}

y = y.map(lambda x: attack_map.get(x, 0))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200)

model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")

print("Model trained successfully")