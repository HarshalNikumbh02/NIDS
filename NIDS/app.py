# ---------------- IMPORTS ----------------
from flask import Flask, request, render_template, redirect, session, jsonify, flash, send_file
import os, io
import pandas as pd
import numpy as np
import joblib
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import bcrypt
from datetime import datetime, timezone, timedelta
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


# Define the IST timezone right below the imports
IST = timezone(timedelta(hours=5, minutes=30))

# ---------------- APP ----------------
app = Flask(__name__)
app.secret_key = os.urandom(24)

# ---------------- DATABASE ----------------
try:
    client = MongoClient(
        "mongodb://localhost:27017/",
        serverSelectionTimeoutMS=5000
    )
    client.admin.command("ping")

    db = client["soc_db"]
    users_collection = db["users"]
    logs_collection = db["logs"]

    # Try to create the unique email index, but don't kill the app if it conflicts
    try:
        users_collection.create_index("email", unique=True)
    except Exception as index_err:
        print("⚠️ Index already exists or conflicted, bypassing...")

    print("✅ MongoDB connected successfully")

except Exception as e:
    users_collection = None
    logs_collection = None
    print(f"❌ MongoDB connection failed. EXACT ERROR: {e}")

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

        elif u2r_score > 0.3:
            final_preds.append(4)

        elif r2l_score > 0.35:
            final_preds.append(3)

        else:
            final_preds.append(int(main_preds[i]))

    return np.array(final_preds)

# ---------------- ROUTES ----------------
@app.route("/")
@app.route("/admin")
def admin():
    if "user" not in session:
        return redirect("/login")
    users_data = list(users_collection.find({}))
    
    return render_template("admin.html", users=users_data)

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

# ---------------- THREAT MITIGATION PLAYBOOKS ----------------
def get_mitigation_playbook(attack_label):
    playbooks = {
        "Normal Traffic": {
            "action": "System operation secure. No anomalous anomalies detected.",
            "commands": "# Continuous passive pattern analysis active.\n# SOC interface operating normally."
        },
        "DoS Attack": {
            "action": "Rate-limit extreme flood frequency. Identify and drop burst pipeline packets immediately.",
            "commands": "sudo iptables -A INPUT -p tcp --dport 80 -m limit --limit 25/minute --limit-burst 100 -j ACCEPT\nsudo iptables -A INPUT -p tcp --syn -j DROP"
        },
        "Probe Attack": {
            "action": "Enforce strict port hiding rules, block network scouting scans, and drop incoming telemetry signatures.",
            "commands": "sudo ufw default deny incoming\nsudo iptables -A INPUT -m psd --psd-weight-threshold 21 --psd-delay-threshold 300 -j DROP"
        },
        "R2L Attack": {
            "action": "Enforce mandatory access management controls, invoke remote pipeline protection rules, and cycle account tokens.",
            "commands": "sudo systemctl restart fail2ban\nsudo iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --set\nsudo iptables -A INPUT -p tcp --dport 22 -m state --state NEW -m recent --update --seconds 60 --hitcount 4 -j DROP"
        },
        "U2R Attack": {
            "action": "Quarantine session instance instantly. Drop horizontal root bypass routes and isolate core assets.",
            "commands": "sudo chmod 700 /usr/bin/su\nsudo pkill -u compromised_session_id\nsudo ausearch -m chroot,priv_change --success -t today"
        }
    }
    return playbooks.get(attack_label, playbooks["Normal Traffic"])

# ---------------- SINGLE PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return jsonify({"success": False, "error": "Session expired. Please log in again."}), 401

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
        playbook = get_mitigation_playbook(label)

        if logs_collection is not None:
            logs_collection.insert_one({
                "user": session.get("user"),
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
                "time":    datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
            })

        # --- SAVE TO SESSION FOR MANUAL PDF REPORT ---
        session['latest_threat'] = {
            "mode": "manual",
            "type": label,
            "status": "SAFE" if status == "safe" else "ATTACK",
            "time": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        }

        return jsonify({
            "success": True,
            "output": label,
            "status": status,
            "playbook": playbook
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    
# ---------------- CSV UPLOAD ----------------
@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    if "user" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"})

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
        
        # Determine main threat class from bulk file for playbook generation
        threat_label = "Normal Traffic"
        if attacks > 0:
            only_attacks = {k: v for k, v in counts.items() if k != "Normal Traffic"}
            threat_label = max(only_attacks, key=only_attacks.get)

        playbook = get_mitigation_playbook(threat_label)

        if logs_collection is not None:
            logs_collection.insert_one({
                "user":    session.get("user"),
                "counts":  counts,
                "total":   total,
                "normal":  normal,
                "attacks": attacks,
                "status":  "SAFE" if normal >= attacks else "ATTACK",
                "time":    datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
            })

        # --- SAVE TO SESSION FOR BULK PDF REPORT ---
        dominant_threat = "Normal Traffic"
        if attacks > 0:
            attack_counts = {k: v for k, v in counts.items() if k != "Normal Traffic"}
            if attack_counts:
                dominant_threat = max(attack_counts, key=attack_counts.get)

        session['latest_threat'] = {
            "mode": "bulk",
            "filename": file.filename,  # <--- NEW: Grabs your exact CSV name
            "total": total,
            "safe": normal,
            "threats": attacks,
            "counts": counts,
            "dominant": dominant_threat,
            "status": "SAFE" if attacks == 0 else "ATTACK",
            "time": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
        }

        return jsonify({
            "success": True,
            "output": f"Total Assessed Instances: {total} | Safe: {normal} | Critical Threats: {attacks}",
            "status": "safe" if normal >= attacks else "danger",
            "summary": counts,
            "playbook": playbook
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

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

            # --- IST TIME FIX ---
            raw_time = item.get("time")
            if isinstance(raw_time, datetime):
                # If it's an old log saved as a MongoDB UTC datetime object, add 5.5 hours to it
                display_time = (raw_time + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
            elif isinstance(raw_time, str):
                # If it's a new log saved as an exact IST string, use it directly
                display_time = raw_time
            else:
                display_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

            logs_data.append({
                "time":   display_time,
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

@app.route('/download_pdf')
def download_pdf():
    threat = session.get('latest_threat')
    if not threat:
        return "No threat data found. Please run a detection first.", 404

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setTitle("NIDS Security Incident Report")
    width, height = letter

    # --- 1. HEADER ---
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "NIDS SECURITY INCIDENT REPORT")
    c.setLineWidth(2)
    c.line(50, height - 60, width - 50, height - 60)

    # --- 2. ALERT STATUS & SUMMARY ---
    if threat['status'] == "SAFE":
        c.setFillColorRGB(0, 0.6, 0) # Green
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 90, "SYSTEM STATUS: CLEAR (NO THREAT)")
    else:
        c.setFillColorRGB(0.8, 0, 0) # Red
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, height - 90, "ALERT: INTRUSION VECTOR DETECTED")

    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 115, f"Timestamp (IST): {threat['time']}")

    # Print Bulk Stats OR Manual Stats
    if threat['mode'] == "bulk":
        c.drawString(50, height - 135, f"Total Assessed Instances: {threat['total']}   |   Safe: {threat['safe']}   |   Critical Threats: {threat['threats']}")
        dominant_attack = threat['dominant']
    else:
        c.drawString(50, height - 135, "Analysis Mode: Single Vector (Manual Input)")
        c.drawString(50, height - 155, f"Classification: {threat['type']}")
        dominant_attack = threat['type']

    # --- 3. MITIGATION PLAYBOOK & COMMANDS ---
    if threat['status'] == "ATTACK":
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, height - 190, "Recommended Response Playbook")

        # Automatically assign the correct commands based on the attack type
        desc = "Quarantine affected system and review network logs."
        cmd = ["sudo ufw enable", "sudo fail2ban-client status"]

        if dominant_attack == "DoS Attack":
            desc = "Drop massive incoming connection spikes and rate-limit IP ranges."
            cmd = ["sudo iptables -A INPUT -p tcp --dport 80 -m limit --limit 25/minute -j ACCEPT", "sudo ufw deny from <ATTACKER_IP>"]
        elif dominant_attack == "Probe Attack":
            desc = "Enforce strict port hiding rules, block network scouting scans, and drop incoming telemetry signatures."
            cmd = ["sudo ufw default deny incoming", "sudo iptables -A INPUT -m psd --psd-weight-threshold 21 --psd-delay-threshold 300 -j DROP"]
        elif dominant_attack in ["R2L Attack", "U2R Attack"]:
            desc = "Immediately revoke compromised SSH keys, terminate active suspicious sessions, and isolate the host."
            cmd = ["sudo pkill -KILL -u <COMPROMISED_USER>", "sudo ufw deny out to any", "sudo passwd -l <COMPROMISED_USER>"]

        c.setFont("Helvetica", 10)
        c.drawString(50, height - 210, desc)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(50, height - 235, "DEPLOYMENT MITIGATION COMMANDS:")

        # Draw a grey rectangle background for the code block
        c.setFillColorRGB(0.95, 0.95, 0.95)
        c.rect(50, height - 295, width - 100, 50, fill=1, stroke=1)
        
        # Write the terminal commands inside the grey box
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Courier", 10)
        y_pos = height - 265
        for line in cmd:
            c.drawString(60, y_pos, line)
            y_pos -= 15

    # --- 4. BULK DATA TABLE (Only draws if a CSV was uploaded) ---
    if threat['mode'] == "bulk":
        start_y = height - 340 if threat['status'] == "ATTACK" else height - 180

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, start_y, "Bulk Threat Assessment Breakdown")

        # Table Header Row (Dark Blue Background)
        start_y -= 20
        c.setFillColorRGB(0.1, 0.2, 0.3)
        c.rect(50, start_y - 15, width - 100, 25, fill=1, stroke=1)
        c.setFillColorRGB(1, 1, 1) # White Text
        c.setFont("Helvetica-Bold", 10)
        c.drawString(70, start_y - 7, "Attack Classification Label")
        c.drawString(320, start_y - 7, "Aggregated Vector Count")

        # Table Data Rows
        c.setFillColorRGB(0, 0, 0) # Black Text
        c.setFont("Helvetica", 10)
        start_y -= 35

        for attack_type, count in threat['counts'].items():
            c.drawString(70, start_y, str(attack_type))
            c.drawString(320, start_y, str(count))
            
            # Draw a subtle line under each row
            c.setLineWidth(0.5)
            c.setFillColorRGB(0.8, 0.8, 0.8)
            c.line(50, start_y - 8, width - 50, start_y - 8)
            c.setFillColorRGB(0, 0, 0) # Reset to black for next text
            
            start_y -= 25

    # --- 5. FOOTER ---
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    c.drawString(50, 40, "Automated forensic report generated by NIDS SOC ML Engine.")

    # There must be EXACTLY ONE c.save() command in this entire function!
    c.save()
    buffer.seek(0)
    
    # Clean up the timestamp for Windows file names
    safe_time = threat['time'].replace(':', '-')
    
    # Create a dynamic file name based on the mode
    if threat['mode'] == 'bulk':
        # Removes '.csv' from the original name so it doesn't end in .csv.pdf
        clean_name = threat.get('filename', 'Dataset').replace('.csv', '')
        final_filename = f"NIDS_Bulk_Report_{clean_name}_{safe_time}.pdf"
    else:
        final_filename = f"NIDS_Manual_Threat_Report_{safe_time}.pdf"

    return send_file(
        buffer, 
        as_attachment=True, 
        download_name=final_filename, 
        mimetype='application/pdf'
    )

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5001)