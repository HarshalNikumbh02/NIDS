# ---------------- IMPORTS ----------------
from flask import Flask, request, render_template, redirect, session, jsonify, flash
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import bcrypt

# ---------------- APP ----------------
app = Flask(__name__)
app.secret_key = "secret123"

# ---------------- DATABASE ----------------
try:
    client = MongoClient(
        "mongodb://127.0.0.1:27017/",
        serverSelectionTimeoutMS=3000,
        connect=True
    )
    client.admin.command("ping")

    db = client["soc_db"]
    users_collection = db["users"]
    logs_collection = db["logs"]

    users_collection.create_index("email", unique=True)
    print("MongoDB connected successfully")

except Exception:
    users_collection = None
    logs_collection = None
    print("MongoDB connection failed")

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
model_r2l = joblib.load("r2l_model.pkl")
model_u2r = joblib.load("u2r_model.pkl")

scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
feature_columns = joblib.load("columns.pkl")

# ---------------- LABELS ----------------
attack_types = {
    0: "Normal Traffic",
    1: "DoS Attack",
    2: "Probe Attack",
    3: "R2L Attack",
    4: "U2R Attack"
}

# ---------------- PREPROCESS ----------------
def preprocess(df):
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_columns]

    for col, enc in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].apply(lambda x: x if x in enc.classes_ else enc.classes_[0])
            df[col] = enc.transform(df[col])

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    return scaler.transform(df)

# ---------------- FINAL PREDICTION ----------------
def predict_final(X, X_raw, columns):

    main_preds = model.predict(X)
    r2l_probs = model_r2l.predict_proba(X)
    u2r_probs = model_u2r.predict_proba(X)

    final_preds = []
    col_index = {col: i for i, col in enumerate(columns)}

    def to_num(val):
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ["yes", "true", "1"]:
                return 1
            if v in ["no", "false", "0"]:
                return 0
            try:
                return float(v)
            except:
                return 0
        return val

    for i in range(len(X)):
        raw = X_raw[i]

        # --- Feature Extraction ---
        logged_in        = to_num(raw[col_index.get("logged_in", 0)])
        failed_logins    = to_num(raw[col_index.get("num_failed_logins", 0)])
        root_shell       = to_num(raw[col_index.get("root_shell", 0)])
        su_attempted     = to_num(raw[col_index.get("su_attempted", 0)])
        num_root         = to_num(raw[col_index.get("num_root", 0)])
        num_compromised  = to_num(raw[col_index.get("num_compromised", 0)])

        count            = to_num(raw[col_index.get("count", 0)])
        serror_rate      = to_num(raw[col_index.get("serror_rate", 0)])
        srv_serror_rate  = to_num(raw[col_index.get("srv_serror_rate", 0)])
        rerror_rate      = to_num(raw[col_index.get("rerror_rate", 0)])

        same_srv         = to_num(raw[col_index.get("same_srv_rate", 0)])
        diff_srv         = to_num(raw[col_index.get("diff_srv_rate", 0)])
        diff_host_rate   = to_num(raw[col_index.get("dst_host_diff_srv_rate", 0)])
        dst_host_same    = to_num(raw[col_index.get("dst_host_same_srv_rate", 0)])
        dst_host_count   = to_num(raw[col_index.get("dst_host_count", 0)])

        u2r_score = u2r_probs[i][1]
        r2l_score = r2l_probs[i][1]

        if (
            (root_shell == 1 and logged_in == 1) or
            (su_attempted == 1 and num_root > 0) or
            (num_compromised > 5 and root_shell == 1) or
            (u2r_score > 0.5)
        ):
            final_preds.append(4)

        # -------- DoS (Denial of Service) --------
        # High connection count + high SYN/REJ error rates
        elif (
            (count > 120 and serror_rate > 0.6) or
            (count > 120 and srv_serror_rate > 0.6) or
            (count > 200 and rerror_rate > 0.6)
        ):
            final_preds.append(1)

        elif (
            (same_srv < 0.4 and diff_srv > 0.5) or

            (diff_host_rate > 0.5 and dst_host_count > 30) or

            (same_srv < 0.3 and count > 5) or

            (diff_srv > 0.7 and count > 3) or

            (dst_host_same < 0.3 and dst_host_count > 20)
        ):
            final_preds.append(2)

        elif (
            (failed_logins >= 1 and logged_in == 0) or
            (failed_logins >= 2) or
            (r2l_score > 0.4 and logged_in == 0) or
            (r2l_score > 0.6)
        ):
            final_preds.append(3)

        # -------- ML Fallback: U2R --------
        elif u2r_score > 0.3:
            final_preds.append(4)

        # -------- ML Fallback: R2L --------
        elif r2l_score > 0.35:
            final_preds.append(3)

        # -------- Main Model Fallback --------
        else:
            final_preds.append(int(main_preds[i]))

    return np.array(final_preds)

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/detect")
def detect():
    if "user" not in session:
        return redirect("/login")
    return render_template("detect.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login")
    return render_template("dashboard.html")

# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        if users_collection is None:
            flash("Database not connected", "danger")
            return redirect("/register")

        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

        try:
            users_collection.insert_one({
                "name": name,
                "email": email,
                "password": hashed_pw
            })
            flash("Registration successful! Please login.", "success")
            return redirect("/login")

        except DuplicateKeyError:
            flash("Email already exists!", "danger")
            return redirect("/register")

    return render_template("register.html")

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        if users_collection is None:
            flash("Database not connected", "danger")
            return redirect("/login")

        user = users_collection.find_one({"email": email})

        if not user:
            flash("Email not registered", "danger")
            return redirect("/login")

        if not bcrypt.checkpw(password.encode(), user["password"]):
            flash("Incorrect password", "danger")
            return redirect("/login")

        session["user"] = user.get("name", "User")
        return redirect("/dashboard")

    return render_template("login.html")

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ---------------- SINGLE PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [v if v != "" else 0 for v in request.form.values()]

        if len(values) < 41:
            values += [0] * (41 - len(values))
        elif len(values) > 41:
            values = values[:41]

        df = pd.DataFrame([values], columns=feature_columns)

        X = preprocess(df)
        X_raw = df.values

        pred = int(predict_final(X, X_raw, feature_columns)[0])
        label = attack_types[pred]
        status = "safe" if label == "Normal Traffic" else "danger"

        if logs_collection is not None:
            logs_collection.insert_one({
                "user": session.get("user", "unknown"),
                "counts": {
                    "Normal Traffic": int(label == "Normal Traffic"),
                    "DoS Attack":     int(label == "DoS Attack"),
                    "Probe Attack":   int(label == "Probe Attack"),
                    "R2L Attack":     int(label == "R2L Attack"),
                    "U2R Attack":     int(label == "U2R Attack"),
                },
                "total":   1,
                "normal":  int(label == "Normal Traffic"),
                "attacks": int(label != "Normal Traffic"),
                "status":  "SAFE" if status == "safe" else "ATTACK",
                "time":    datetime.utcnow()
            })

        return render_template("detect.html", output=label, status=status)

    except Exception as e:
        return render_template("detect.html", output=str(e), status="danger")

# ---------------- CSV UPLOAD ----------------
@app.route("/upload_csv", methods=["POST"])
def upload_csv():

    if "user" not in session:
        return redirect("/login")

    try:
        file = request.files["file"]
        df = pd.read_csv(file, header=None)

        if df.shape[1] > 41:
            df = df.iloc[:, :41]
        else:
            for _ in range(41 - df.shape[1]):
                df[df.shape[1]] = 0

        df.columns = feature_columns

        X = preprocess(df)
        X_raw = df.values

        preds = predict_final(X, X_raw, feature_columns)

        counts = {v: 0 for v in attack_types.values()}
        for p in preds:
            counts[attack_types[int(p)]] += 1

        total  = sum(counts.values())
        normal = counts["Normal Traffic"]
        attacks = total - normal

        if logs_collection is not None:
            logs_collection.insert_one({
                "user":    session.get("user", "unknown"),
                "counts":  counts,
                "total":   total,
                "normal":  normal,
                "attacks": attacks,
                "status":  "SAFE" if normal >= attacks else "ATTACK",
                "time":    datetime.utcnow()
            })

        return render_template(
            "detect.html",
            output=f"Total: {total} | Normal: {normal} | Attacks: {attacks}",
            status="safe" if normal >= attacks else "danger",
            summary=counts
        )

    except Exception as e:
        return render_template("detect.html", output=str(e), status="danger")

# ---------------- DASHBOARD API ----------------
@app.route("/api/data")
def api_data():

    if "user" not in session:
        return jsonify({"total": 0, "normal": 0, "attacks": 0, "attack_stats": {}, "logs": []})

    total  = 0
    normal = 0

    attack_stats = {
        "Normal": 0,
        "DoS":    0,
        "Probe":  0,
        "R2L":    0,
        "U2R":    0
    }

    logs_data = []

    if logs_collection is not None:

        data = list(
            logs_collection.find({"user": session.get("user")})
            .sort("time", -1)
            .limit(100)
        )

        for item in data:
            c = item.get("counts", {})

            total  += sum(c.values())
            normal += c.get("Normal Traffic", 0)

            attack_stats["Normal"] += c.get("Normal Traffic", 0)
            attack_stats["DoS"]    += c.get("DoS Attack", 0)
            attack_stats["Probe"]  += c.get("Probe Attack", 0)
            attack_stats["R2L"]    += c.get("R2L Attack", 0)
            attack_stats["U2R"]    += c.get("U2R Attack", 0)

        for item in data[:10]:
            c = item.get("counts", {})
            attack = max(c, key=c.get) if c else "Normal Traffic"

            logs_data.append({
                "time":   item.get("time", datetime.utcnow()).strftime("%Y-%m-%d %H:%M:%S"),
                "attack": attack,
                "status": "Blocked" if attack != "Normal Traffic" else "Allowed"
            })

    return jsonify({
        "total":        total,
        "normal":       normal,
        "attacks":      total - normal,
        "attack_stats": attack_stats,
        "logs":         logs_data
    })

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)