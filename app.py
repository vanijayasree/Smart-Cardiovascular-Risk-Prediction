from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import os
import csv
import shap
import matplotlib.pyplot as plt
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
import io
import base64
from io import BytesIO
import seaborn as sns
import pandas as pd

from config import Config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def plot_to_img(fig):
    """Convert Matplotlib figure to base64 image for embedding."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

# -------------------------
# Flask Config
# -------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = Config.SECRET_KEY

# -------------------------
# Database Connection
# -------------------------
def get_db():
    return mysql.connector.connect(
        host=Config.DB_HOST,
        user=Config.DB_USER,
        password=Config.DB_PASSWORD,
        database=Config.DB_NAME
    )

# -------------------------
# Init DB (adds missing cols + system_settings table)
# -------------------------
def init_db():
    conn = get_db()
    cursor = conn.cursor()
    # add role if not exists
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN role VARCHAR(20) DEFAULT 'User'")
    except mysql.connector.Error as e:
        if e.errno != 1060:  # 1060 = duplicate column name, expected/harmless
            print(f"init_db warning (role column): {e}")
    # add is_active if not exists
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT 1")
    except mysql.connector.Error as e:
        if e.errno != 1060:
            print(f"init_db warning (is_active column): {e}")
    # add shap_filename column to heart_reports if not exists (for explainability charts)
    try:
        cursor.execute("ALTER TABLE heart_reports ADD COLUMN shap_filename VARCHAR(255)")
    except mysql.connector.Error as e:
        if e.errno != 1060:
            print(f"init_db warning (shap_filename column): {e}")
    # create system settings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_settings (
            id INT PRIMARY KEY AUTO_INCREMENT,
            maintenance VARCHAR(10) DEFAULT 'off',
            allow_register VARCHAR(10) DEFAULT 'yes',
            theme VARCHAR(20) DEFAULT 'light'
        )
    """)
    cursor.execute("SELECT COUNT(*) FROM system_settings")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO system_settings (maintenance, allow_register, theme) VALUES ('off','yes','light')")
    conn.commit()
    cursor.close()
    conn.close()

init_db()

# -------------------------
# Load ML Model
# -------------------------
models = joblib.load(os.path.join(BASE_DIR, "models", "heart_model.pkl"))     # adjust path to where your model is saved
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

# -------------------------
# SHAP Explainability Setup
# -------------------------
FEATURE_NAMES = [
    "Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
    "Fasting Blood Sugar", "Resting ECG", "Max Heart Rate",
    "Exercise Angina", "Oldpeak", "ST Slope"
]

# The trained model is a CalibratedClassifierCV wrapping RandomForest models
# (one per CV fold). shap.TreeExplainer works on each underlying RandomForest;
# we average across folds for a stable per-prediction explanation.
try:
    _shap_tree_explainers = [
        shap.TreeExplainer(cc.estimator) for cc in models.calibrated_classifiers_
    ]
except Exception as e:
    print("SHAP setup warning:", e)
    _shap_tree_explainers = []

SHAP_DIR = os.path.join(BASE_DIR, "static", "shap_reports")
os.makedirs(SHAP_DIR, exist_ok=True)


def generate_shap_explanation(scaled_row, report_id):
    """Generates a bar chart showing which features pushed this specific
    prediction toward higher or lower heart-disease risk. Returns the
    filename (relative to /static) to store + render, or None on failure."""
    if not _shap_tree_explainers:
        print("WARNING: No SHAP explainers available")
        return None
    try:
        all_shap_vals = []
        for explainer in _shap_tree_explainers:
            sv = explainer.shap_values(scaled_row)
            # sv shape can be (1, n_features, n_classes) or a list [class0, class1]
            if isinstance(sv, list):
                class1 = np.array(sv[1])[0]
            else:
                sv = np.array(sv)
                class1 = sv[0, :, 1] if sv.ndim == 3 else sv[0]
            all_shap_vals.append(class1)

        avg_shap = np.mean(all_shap_vals, axis=0)

        order = np.argsort(np.abs(avg_shap))[::-1]
        top_features = [FEATURE_NAMES[i] for i in order]
        top_values = [avg_shap[i] for i in order]
        colors_list = ["#d9534f" if v > 0 else "#5cb85c" for v in top_values]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top_features[::-1], top_values[::-1], color=colors_list[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Impact on predicted risk (SHAP value)")
        ax.set_title("What influenced this prediction")
        plt.tight_layout()

        filename = f"shap_reports/shap_{report_id}.png"
        full_path = os.path.join(BASE_DIR, "static", filename)
        fig.savefig(full_path, dpi=120)
        plt.close(fig)
        print(f"✓ SHAP chart saved: {full_path}")
        return filename
    except Exception as e:
        import traceback
        print(f"✗ SHAP generation error: {e}")
        traceback.print_exc()
        return None

# -------------------------
# Routes
# -------------------------

@app.route('/')
def index():
    return render_template('index.html')

# -------------------------
# Register
# -------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    """
    User registration route.
    SECURITY FIX: Role is HARDCODED as 'User' - users cannot select their role
    Only setupp.py can create the initial admin account
    """
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT * FROM system_settings LIMIT 1")
    settings = cursor.fetchone()
    if settings and settings['allow_register'] == 'no':
        flash("Registrations are disabled by admin.", "danger")
        return redirect(url_for('login'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        # SECURITY FIX: Do NOT accept role from form!
        # role = request.form['role']  # ← DELETED - Users cannot select their role
        
        # Input validation
        if not username or not password:
            flash("Username and password are required", "danger")
            return redirect(url_for('register'))
        
        if len(username) < 3:
            flash("Username must be at least 3 characters", "danger")
            return redirect(url_for('register'))
        
        if len(password) < 6:
            flash("Password must be at least 6 characters", "danger")
            return redirect(url_for('register'))
        
        hashed = generate_password_hash(password)

        try:
            # SECURITY FIX: Always hardcode 'User' role - never accept from user input
            cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (%s,%s,%s)",
                           (username, hashed, 'User'))  # ← 'User' is hardcoded, not from form
            conn.commit()
            flash("Account created! Please login.", "success")
            return redirect(url_for('login'))
        except mysql.connector.Error as err:
            if err.errno == 1062:  # Duplicate entry error code
                flash("Username already exists", "danger")
            else:
                flash(f"Registration failed: {err}", "danger")
                print("REGISTER ERROR:", err)
        
        finally:
            conn.close()

    return render_template('register.html')

# -------------------------
# Login
# -------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cursor.fetchone()

        if user and check_password_hash(user['password_hash'], password):
            if not user['is_active']:
                flash("Account disabled. Contact admin.", "danger")
                return redirect(url_for('login'))

            session['username'] = user['username']
            session['role'] = user['role']

            if user['role'] == 'Admin':
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('user_dashboard'))
        else:
            flash("Invalid credentials", "danger")

    return render_template('login.html')

# -------------------------
# Logout
# -------------------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/about')
def about():
    return render_template('about.html')


# -------------------------
# User Dashboard
# -------------------------
@app.route('/user')
def user_dashboard():
    if 'username' not in session or session.get('role') != 'User':
        return redirect(url_for('login'))
    return render_template('user_dashboard.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    patient_name = request.form.get('patient_name', '').strip()

    # --------------------------
    # Input validation
    # --------------------------
    # Field name -> (label shown to user, min, max, type)
    FIELD_RULES = {
        'age':             ('Age', 1, 120, int),
        'sex':             ('Sex', 0, 1, int),
        'chest_pain_type': ('Chest Pain Type', 0, 3, int),
        'resting_bp':      ('Resting BP', 60, 250, int),
        'cholesterol':     ('Cholesterol', 50, 700, int),
        'fasting_bs':      ('Fasting Blood Sugar', 0, 1, int),
        'rest_ecg':        ('Resting ECG', 0, 2, int),
        'max_hr':          ('Max Heart Rate', 40, 250, int),
        'exercise_angina': ('Exercise Angina', 0, 1, int),
        'oldpeak':         ('Oldpeak', -3, 10, float),
        'st_slope':        ('ST Slope', 0, 2, int),
    }

    if not patient_name:
        flash("Please enter the patient's name.", "danger")
        return redirect(url_for('user_dashboard'))
    if len(patient_name) > 100:
        flash("Patient name is too long (max 100 characters).", "danger")
        return redirect(url_for('user_dashboard'))

    values = {}
    for field, (label, min_val, max_val, caster) in FIELD_RULES.items():
        raw = request.form.get(field, '').strip()
        if raw == '':
            flash(f"Please fill in '{label}'.", "danger")
            return redirect(url_for('user_dashboard'))
        try:
            value = caster(raw)
        except ValueError:
            flash(f"'{label}' must be a valid number.", "danger")
            return redirect(url_for('user_dashboard'))
        if not (min_val <= value <= max_val):
            flash(f"'{label}' must be between {min_val} and {max_val}.", "danger")
            return redirect(url_for('user_dashboard'))
        values[field] = value

    # --------------------------
    # Collect features from form (now validated)
    # --------------------------
    data = [
        values['age'],
        values['sex'],
        values['chest_pain_type'],
        values['resting_bp'],
        values['cholesterol'],
        values['fasting_bs'],
        values['rest_ecg'],
        values['max_hr'],
        values['exercise_angina'],
        values['oldpeak'],
        values['st_slope'],
    ]

    # --------------------------
    # Scale and predict
    # --------------------------
    scaled = scaler.transform([data])
    pred = models.predict(scaled)[0]
    prob = models.predict_proba(scaled)[0][1]  # probability of the positive (disease) class

    # Risk category (3-tier: Low / Borderline / High)
    if prob < 0.40:
        risk = "Low Risk"
    elif prob <= 0.55:
        risk = "Borderline Risk"
    else:
        risk = "High Risk"

    # --------------------------
    # Personalized Recommendations
    # --------------------------
    cholesterol = data[4]
    max_hr = data[7]
    resting_bp = data[3]

    # Diet recommendations — now considers BOTH overall risk level and cholesterol,
    # so a High/Borderline Risk patient never gets a "maintain current diet" message
    # just because their cholesterol happens to be normal.
    if risk == "High Risk" or cholesterol > 240:
        diet = "Low-fat, high-fiber diet. Avoid red meat and fried food."
        diet_details = [
            "Bananas, apples, oranges (low sodium fruits)",
            "Oats, lentils, beans (high fiber)",
            "Olive oil, salmon, walnuts (healthy fats)",
            "Avoid: fried food, butter, cheese, red meat"
        ]
    elif risk == "Borderline Risk" or cholesterol > 200:
        diet = "Balanced diet with limited fatty food."
        diet_details = [
            "Whole grains (brown rice, oats)",
            "Leafy greens (spinach, broccoli)",
            "Lean proteins (chicken, tofu, legumes)",
            "Avoid: processed snacks, sugary drinks"
        ]
    else:
        diet = "Maintain current healthy diet."
        diet_details = [
            "Continue with fruits, vegetables, and whole grains",
            "Regular hydration (8–10 glasses of water)",
            "Include nuts & seeds in moderation",
            "Limit alcohol & processed foods"
        ]

    # Exercise recommendations — considers overall risk level too, not just max heart rate
    if risk == "High Risk" or max_hr < 120:
        exercise = "Light walking and yoga. Avoid high-intensity workouts."
        exercise_details = [
            "20–30 mins walking daily",
            "Basic yoga stretches",
            "Light cycling at slow pace",
            "Avoid: running, heavy gym workouts"
        ]
    else:
        exercise = "30 mins of jogging, cycling, or swimming."
        exercise_details = [
            "Jogging or brisk walking (30 mins)",
            "Cycling (20–25 mins)",
            "Swimming (gentle laps)",
            "Yoga + breathing exercises"
        ]

    # Stress management — considers overall risk level too, not just resting BP
    if risk == "High Risk" or resting_bp > 140:
        stress = "Daily meditation, breathing exercises, and reduce salt intake."
        stress_details = [
            "Mindfulness meditation (10 mins)",
            "Deep breathing exercises",
            "Reduce salt & caffeine intake",
            "Doctor checkups every 2–3 months"
        ]
    else:
        stress = "Mindfulness & regular relaxation activities."
        stress_details = [
            "Listening to calm music",
            "Walking in nature",
            "7–8 hours of sleep",
            "Avoid smoking/alcohol"
        ]

    # --------------------------
    # Save into DB
    # --------------------------
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(""" 
        INSERT INTO heart_reports
        (patient_name, age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs,
         rest_ecg, max_hr, exercise_angina, oldpeak, st_slope, prediction, probability,
         diet_plan, exercise_plan, stress_plan, created_by)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (patient_name, *data,
          risk, float(prob), diet, exercise, stress, session['username']))
    report_id = cursor.lastrowid

    # Generate SHAP explanation chart for this specific prediction
    shap_filename = generate_shap_explanation(scaled, report_id)
    print(f"DEBUG: shap_filename generated = {shap_filename}")  # Debug line
    if shap_filename:
        cursor.execute("UPDATE heart_reports SET shap_filename=%s WHERE id=%s", (shap_filename, report_id))
        print(f"DEBUG: updated DB with shap_filename")  # Debug line

    conn.commit()

    # --------------------------
    # Render Dashboard with details
    # --------------------------
    print(f"DEBUG: passing to template shap_filename={shap_filename}")  # Debug line
    return render_template(
        "recommendations.html",
        prediction=risk,
        probability=round(prob * 100, 2),
        name=patient_name,
        report_id=report_id,
        diet=diet, diet_details=diet_details,
        exercise=exercise, exercise_details=exercise_details,
        stress=stress, stress_details=stress_details,
        shap_filename=shap_filename
    )

# ===================== SHOW RECOMMENDATIONS =====================

@app.route('/show_recommendations/<int:report_id>')
def show_recommendations(report_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM heart_reports WHERE id = %s", (report_id,))
    report = cursor.fetchone()

    if not report:
        flash("Report not found!", "danger")
        return redirect(url_for("user_dashboard"))

    return render_template(
        "recommendations.html",
        report_id=report_id,  # 👈 Pass report_id to template
        name=report["patient_name"],
        prediction=report["prediction"],
        probability=round(report["probability"] * 100, 2),
        diet=report["diet_plan"],
        exercise=report["exercise_plan"],
        stress=report["stress_plan"],
        shap_filename=report.get("shap_filename")
    )

@app.route('/show_evaluation/<int:report_id>')
def show_evaluation(report_id):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    import io, base64

    # --- Load dataset ---
    df = pd.read_csv(os.path.join(BASE_DIR, "heart_clean.csv"))  # adjust to your dataset path
    if "S.NO" in df.columns:
        df = df.drop("S.NO", axis=1)

    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "Accuracy": round(accuracy_score(y_test, y_pred), 3),
            "Precision": round(precision_score(y_test, y_pred), 3),
            "Recall": round(recall_score(y_test, y_pred), 3),
            "F1-Score": round(f1_score(y_test, y_pred), 3),
            "ROC-AUC": round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 3)
        }

    results_df = pd.DataFrame(results).T

    # --- Feature Importance (Random Forest) ---
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    # Plot Feature Importance
    plt.figure(figsize=(8,6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
    plt.title("Feature Importance (Random Forest)")
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template(
        "evaluation.html",
        results=results_df.to_html(classes="table table-bordered", float_format="%.3f"),
        plot_url=plot_url,
        report_id=report_id
    )


# ===================== SHOW DIET =====================
@app.route('/show_diet/<int:report_id>')
def show_diet(report_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT diet_plan, prediction FROM heart_reports WHERE id = %s", (report_id,))
    report = cursor.fetchone()

    if not report:
        flash("Diet plan not found!", "danger")
        return redirect(url_for("user_dashboard"))

    diet_plan = report["diet_plan"]
    prediction = report["prediction"]

    # Example detailed foods based on risk + diet type
    if prediction.lower() == "high risk":
        if "low-fat" in diet_plan.lower():
            diet_details = ["Oats", "Brown rice", "Leafy greens(Spinach, Mustard Greens, Swiss Chard)", "Lentils", "Grilled fish"]
        elif "balanced" in diet_plan.lower():
            diet_details = ["Whole grains", "Lean chicken", "Vegetables(Carrots,Tomatoes, Cauliflower,Green Beans )", "Fruits(Berries,Apples,Pomegranates,Grapes)", "Olive oil"]
        else:
            diet_details = ["Berries", "Broccoli", "Walnuts", "Avocados", "Green tea"]
    else:  # Low Risk
        diet_details = ["Fruits(Bananas,Oranges & Citrus Fruits)", "Vegetables(Cucumber,Spinach & Kale,Bell Peppers)", "Nuts", "Plenty of water", "Occasional fish/chicken"]

    return render_template("diet.html",
                           diet=diet_plan,
                           diet_details=diet_details,
                           prediction=prediction,
                           report_id=report_id)



# ===================== SHOW EXERCISE =====================
@app.route('/show_exercise/<int:report_id>')
def show_exercise(report_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT exercise_plan, prediction FROM heart_reports WHERE id = %s", (report_id,))
    report = cursor.fetchone()

    if not report:
        flash("Exercise plan not found!", "danger")
        return redirect(url_for("user_dashboard"))

    exercise_plan = report["exercise_plan"]
    prediction = report["prediction"]

    if prediction.lower() == "high risk":
        if "yoga" in exercise_plan.lower():
            exercise_details = ["Breathing exercises", "Gentle stretching", "Slow walking"]
        else:
            exercise_details = ["Walking (20 mins)", "Light cycling", "Chair yoga"]
    else:  # Low Risk
        exercise_details = ["Jogging (30 mins)", "Cycling", "Swimming", "Strength training"]

    return render_template("exercise.html",
                           exercise=exercise_plan,
                           exercise_details=exercise_details,
                           prediction=prediction,
                           report_id=report_id)


# ===================== SHOW STRESS =====================
@app.route('/show_stress/<int:report_id>')
def show_stress(report_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT stress_plan, prediction FROM heart_reports WHERE id = %s", (report_id,))
    report = cursor.fetchone()

    if not report:
        flash("Stress plan not found!", "danger")
        return redirect(url_for("user_dashboard"))

    stress_plan = report["stress_plan"]
    prediction = report["prediction"]

    if prediction.lower() == "high risk":
        if "meditation" in stress_plan.lower():
            stress_details = ["15 min meditation", "Deep breathing", "Yoga Nidra", "Progressive relaxation"]
        else:
            stress_details = ["Listening to calm music", "Light walks in nature", "Counseling sessions"]
    else:  # Low Risk
        stress_details = ["Listening to music", "Walking outdoors", "Socializing with friends", "Mindfulness journaling"]

    return render_template("stress.html",
                           stress=stress_plan,
                           stress_details=stress_details,
                           prediction=prediction,
                           report_id=report_id)


# ===================== DOWNLOAD REPORT =====================

@app.route('/download_report/<int:report_id>')
def download_report(report_id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM heart_reports WHERE id = %s", (report_id,))
    report = cursor.fetchone()

    if not report:
        flash("Report not found!", "danger")
        return redirect(url_for("user_dashboard"))

    buffer = io.BytesIO()

    # -------------------- Mapping dictionaries --------------------
    sex_map = {0: "Female", 1: "Male"}
    cp_map = {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal Pain", 3: "Asymptomatic"}
    fbs_map = {0: "Fasting BS < 120 mg/dl", 1: "Fasting BS ≥ 120 mg/dl"}
    ecg_map = {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}
    exang_map = {0: "No", 1: "Yes"}
    slope_map = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}

    # Convert raw values to readable text
    sex = sex_map.get(report.get("sex"), report.get("sex"))
    cp = cp_map.get(report.get("chest_pain_type"), report.get("chest_pain_type"))
    fbs = fbs_map.get(report.get("fasting_bs"), report.get("fasting_bs"))
    ecg = ecg_map.get(report.get("rest_ecg"), report.get("rest_ecg"))
    exang = exang_map.get(report.get("exercise_angina"), report.get("exercise_angina"))
    slope = slope_map.get(report.get("st_slope"), report.get("st_slope"))

    # -------------------- PDF Setup --------------------
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=80,
        bottomMargin=60,
    )

    styles = getSampleStyleSheet()
    body = ParagraphStyle("Body", parent=styles["Normal"], fontName="Helvetica", fontSize=10, leading=14)
    head = ParagraphStyle("Head", parent=styles["Heading5"], fontName="Helvetica-Bold", fontSize=11,
                          textColor=colors.whitesmoke, alignment=1)
    title = ParagraphStyle("TitleBig", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, alignment=1)
    info = ParagraphStyle("Info", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=14)

    story = []

    # -------------------- HEADER & FOOTER --------------------
    from reportlab.lib.utils import ImageReader

    def header_footer(canvas, doc):
        canvas.saveState()
        page_width, page_height = doc.pagesize

        # -------- Background Watermark (centered) --------
        try:
            bg_path = os.path.join(BASE_DIR, "static", "images", "watermark.jpeg")
            watermark = ImageReader(bg_path)
            img_width, img_height = watermark.getSize()
            scale = 0.4
            wm_width = img_width * scale
            wm_height = img_height * scale
            x = (page_width - wm_width) / 2
            y = (page_height - wm_height) / 2
            canvas.drawImage(watermark, x, y, width=wm_width, height=wm_height, mask="auto", preserveAspectRatio=True)
        except Exception as e:
            print("Watermark error:", e)

        # -------- Left Logo (Aurora) --------
        try:
            aurora_logo = os.path.join(BASE_DIR, "static", "images", "generic_logo.png")
            canvas.drawImage(
                aurora_logo,
                40, page_height - 80,  # lower Y to align properly
                width=90, height=85,  # increased size
                preserveAspectRatio=True,
                mask="auto"
            )
        except Exception as e:
            print(f"Left logo error: {e}")
        try:
            right_logo = os.path.join(BASE_DIR, "static", "images", "logo5.png")
            canvas.drawImage(
                right_logo,
                page_width - 120, page_height - 80,  # right side positioning
                width=90, height=90,  # slightly bigger
                preserveAspectRatio=True,
                mask="auto"
            )
        except Exception as e:
            print(f"Right logo error: {e}")
        canvas.setFont("Helvetica-Bold", 16)
        canvas.setFillColor(colors.HexColor("#0d6efd"))  # Blue shade
        canvas.drawCentredString(page_width / 2, page_height - 40, "Cardiovascular Risk Prediction Report")

        # -------- Footer --------
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(40, 40, "Cardiovascular Risk Prediction Report")
        canvas.drawRightString(page_width - 40, 40, f"Page {doc.page}")

        canvas.restoreState()

    # -------------------- Report Content --------------------
    story.append(Paragraph("&#9632; Heart Analysis Report", title))
    story.append(Spacer(1, 20))

    # Patient Info
    patient_details = f"""
    <b>Patient Name:</b> {report['patient_name']} <br/>
    <b>Age:</b> {report.get('age', 'N/A')} <br/>
    <b>Sex:</b> {sex} <br/>
    """
    story.append(Paragraph(patient_details, info))
    story.append(Spacer(1, 12))

    # -------------------- Clinical Inputs Table --------------------
    story.append(Paragraph("<b>Patient Clinical Inputs</b>", styles["Heading3"]))
    story.append(Spacer(1, 10))

    clinical_data = [
        ["Chest Pain Type", cp],
        ["Resting BP (mmHg)", report.get("resting_bp", "N/A")],
        ["Cholesterol (mg/dl)", report.get("cholesterol", "N/A")],
        ["Fasting BS", fbs],
        ["Resting ECG", ecg],
        ["Max HR (bpm)", report.get("max_hr", "N/A")],
        ["Exercise Angina", exang],
        ["Oldpeak", report.get("oldpeak", "N/A")],
        ["ST Slope", slope],
    ]

    clinical_tbl = Table(clinical_data, colWidths=[0.35 * doc.width, 0.65 * doc.width])
    clinical_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#dbeafe")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#0f172a")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),

        ("TEXTCOLOR", (1, 0), (1, -1), colors.black),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),

        ("ROWBACKGROUNDS", (1, 0), (1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.75, colors.grey),
    ]))
    story.append(clinical_tbl)
    story.append(Spacer(1, 16))

    # -------------------- Prediction Result --------------------
    story.append(Paragraph("<b>Prediction Result</b>", styles["Heading3"]))
    risk_level = report["prediction"]
    risk_color = "green" if risk_level.lower() == "low risk" else "red"
    risk_html = f"""
    <b>Risk Level:</b> <font color='{risk_color}'>{risk_level}</font><br/>
    <b>Probability:</b> {round(report['probability'] * 100, 2)}%
    """
    story.append(Spacer(1, 8))
    story.append(Paragraph(risk_html, body))
    story.append(Spacer(1, 16))

    # -------------------- Recommendations Table --------------------
    story.append(Paragraph("<b>Personalized Recommendations</b>", styles["Heading3"]))
    width = doc.width
    col_widths = [0.20 * width, 0.35 * width, 0.45 * width]

    table_data = [
        [Paragraph("Category", head), Paragraph("Recommendation", head), Paragraph("Examples", head)],
        [
            "Diet",
            Paragraph(report["diet_plan"], body),
            Paragraph("Fruits, Vegetables, Nuts, Olive oil", body),
        ],
        [
            "Exercise",
            Paragraph(report["exercise_plan"], body),
            Paragraph("Walking, Jogging, Yoga", body),
        ],
        [
            "Stress Mgmt",
            Paragraph(report["stress_plan"], body),
            Paragraph("Meditation, Breathing, Counseling", body),
        ],
    ]

    tbl = Table(table_data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d6efd")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.8, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)

    # -------------------- Build PDF --------------------
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"Heart_Report_{report_id}.pdf",
        mimetype="application/pdf",
    )

@app.route('/user/visualizations')
def user_visualizations():
    if 'username' not in session or session.get('role') != 'User':
        return redirect(url_for('login'))

    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT age, cholesterol, resting_bp, prediction, probability FROM heart_reports WHERE created_by=%s", (session['username'],))
    rows = cursor.fetchall()

    if not rows:
        flash("No reports available for visualization!", "warning")
        return redirect(url_for("user_dashboard"))

    df = pd.DataFrame(rows)
    plots = {}

    # Risk Distribution
    fig, ax = plt.subplots()
    sns.countplot(x="prediction", data=df, palette="Set2", ax=ax)
    ax.set_title("Risk Distribution")
    plots["Risk Distribution"] = plot_to_img(fig)

    # Age vs Risk
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="age", hue="prediction", multiple="stack", bins=10, ax=ax)
    ax.set_title("Age vs Risk Level")
    plots["Age vs Risk"] = plot_to_img(fig)

    # Cholesterol Distribution
    fig, ax = plt.subplots()
    sns.boxplot(x="prediction", y="cholesterol", data=df, palette="Set3", ax=ax)
    ax.set_title("Cholesterol by Risk Level")
    plots["Cholesterol Distribution"] = plot_to_img(fig)

    # Probability Distribution
    fig, ax = plt.subplots()
    sns.histplot(df["probability"], bins=10, kde=True, ax=ax, color="purple")
    ax.set_title("Prediction Probability Distribution")
    plots["Probability Distribution"] = plot_to_img(fig)

    # Resting BP vs Risk
    fig, ax = plt.subplots()
    sns.boxplot(x="prediction", y="resting_bp", data=df, palette="coolwarm", ax=ax)
    ax.set_title("Resting BP by Risk Level")
    plots["Resting BP vs Risk"] = plot_to_img(fig)

    # Age Distribution (Pie Chart)
    fig, ax = plt.subplots()

    # Define custom bins
    age_bins = pd.cut(df["age"], bins=[0, 20, 45, 65, 100], right=False)

    # Count frequencies
    age_counts = age_bins.value_counts().sort_index()

    # Plot pie chart
    ax.pie(age_counts, labels=age_counts.index.astype(str), autopct='%1.1f%%', startangle=90)
    ax.set_title("Age Distribution")

    # Save to plots dictionary
    plots["Age Distribution"] = plot_to_img(fig)

    # Cholesterol vs Resting BP (Scatter)
    fig, ax = plt.subplots()
    sns.scatterplot(x="cholesterol", y="resting_bp", hue="prediction", data=df, palette="Set1", ax=ax)
    ax.set_title("Cholesterol vs Resting BP")
    plots["Cholesterol vs Resting BP"] = plot_to_img(fig)

    # Probability vs Cholesterol (Scatter)
    fig, ax = plt.subplots()
    sns.scatterplot(x="cholesterol", y="probability", hue="prediction", data=df, palette="Dark2", ax=ax)
    ax.set_title("Probability vs Cholesterol")
    plots["Probability vs Cholesterol"] = plot_to_img(fig)

    return render_template("visualizations.html", plots=plots, role="User")



# User reports
@app.route('/my_reports')
def my_reports():
    if 'username' not in session or session.get('role') != 'User':
        return redirect(url_for('login'))
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM heart_reports WHERE created_by=%s", (session['username'],))
    reports = cursor.fetchall()
    return render_template('my_reports.html', reports=reports)

# -------------------------
# Admin Dashboard
# -------------------------
@app.route('/admin')
def admin_dashboard():
    if 'username' not in session or session.get('role') != 'Admin':
        return redirect(url_for('login'))

    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM heart_reports ORDER BY created_at DESC")
    reports = cursor.fetchall()
    cursor.execute("SELECT id, username, role, is_active FROM users")
    users = cursor.fetchall()
    cursor.execute("SELECT * FROM system_settings LIMIT 1")
    settings = cursor.fetchone()

    return render_template('admin_dashboard.html', reports=reports, users=users, settings=settings)

# Delete report
@app.route('/report/delete/<int:id>', methods=['POST'])
def delete_report(id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM heart_reports WHERE id=%s", (id,))
    conn.commit()
    return jsonify(ok=True)

# Export CSV
@app.route('/export')
def export_reports():
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM heart_reports")
    rows = cursor.fetchall()

    path = "exported_reports.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return "Reports exported to exported_reports.csv"

# Delete User
@app.route('/admin/user/delete/<int:id>', methods=['POST'])
def delete_user(id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id=%s", (id,))
    conn.commit()
    return jsonify(ok=True)

# Toggle User active/inactive
@app.route('/admin/user/toggle/<int:id>', methods=['POST'])
def toggle_user(id):
    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT is_active FROM users WHERE id=%s", (id,))
    u = cursor.fetchone()
    if u:
        new_status = 0 if u['is_active'] else 1
        cursor.execute("UPDATE users SET is_active=%s WHERE id=%s", (new_status, id))
        conn.commit()
        return jsonify(ok=True)
    return jsonify(ok=False)



# Update settings
@app.route('/admin/settings', methods=['POST'])
def update_settings():
    maintenance = 'on' if request.form.get('maintenance') else 'off'
    allow_register = 'yes' if request.form.get('allow_register') else 'no'
    theme = request.form.get('theme')

    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE system_settings SET maintenance=%s, allow_register=%s, theme=%s WHERE id=1",
                   (maintenance, allow_register, theme))
    conn.commit()
    flash("Settings updated!", "success")
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/visualizations')
def admin_visualizations():
    if 'username' not in session or session.get('role') != 'Admin':
        return redirect(url_for('login'))

    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT age, cholesterol, resting_bp, prediction, probability FROM heart_reports")
    rows = cursor.fetchall()

    if not rows:
        flash("No reports available for visualization!", "warning")
        return redirect(url_for("admin_dashboard"))

    df = pd.DataFrame(rows)
    plots = {}

    # Risk Distribution
    fig, ax = plt.subplots()
    sns.countplot(x="prediction", data=df, palette="Set1", ax=ax)
    ax.set_title("Risk Distribution")
    plots["Risk Distribution"] = plot_to_img(fig)

    # Age vs Risk
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="age", hue="prediction", multiple="stack", bins=10, ax=ax)
    ax.set_title("Age vs Risk Level")
    plots["Age vs Risk"] = plot_to_img(fig)

    # Cholesterol Distribution
    fig, ax = plt.subplots()
    sns.boxplot(x="prediction", y="cholesterol", data=df, palette="Set2", ax=ax)
    ax.set_title("Cholesterol by Risk Level")
    plots["Cholesterol Distribution"] = plot_to_img(fig)

    # Probability Distribution
    fig, ax = plt.subplots()
    sns.histplot(df["probability"], bins=10, kde=True, ax=ax, color="orange")
    ax.set_title("Prediction Probability Distribution")
    plots["Probability Distribution"] = plot_to_img(fig)

    # Resting BP vs Risk
    fig, ax = plt.subplots()
    sns.boxplot(x="prediction", y="resting_bp", data=df, palette="coolwarm", ax=ax)
    ax.set_title("Resting BP by Risk Level")
    plots["Resting BP vs Risk"] = plot_to_img(fig)

    # Age Distribution (Pie Chart)
    fig, ax = plt.subplots()

    # Define custom bins
    age_bins = pd.cut(df["age"], bins=[0, 20, 45, 65, 100], right=False)

    # Count frequencies
    age_counts = age_bins.value_counts().sort_index()

    # Plot pie chart
    ax.pie(age_counts, labels=age_counts.index.astype(str), autopct='%1.1f%%', startangle=90)
    ax.set_title("Age Distribution")

    # Save to plots dictionary
    plots["Age Distribution"] = plot_to_img(fig)

    # Cholesterol vs Resting BP (Scatter)
    fig, ax = plt.subplots()
    sns.scatterplot(x="cholesterol", y="resting_bp", hue="prediction", data=df, palette="Set1", ax=ax)
    ax.set_title("Cholesterol vs Resting BP")
    plots["Cholesterol vs Resting BP"] = plot_to_img(fig)

    # Probability vs Cholesterol (Scatter)
    fig, ax = plt.subplots()
    sns.scatterplot(x="cholesterol", y="probability", hue="prediction", data=df, palette="Dark2", ax=ax)
    ax.set_title("Probability vs Cholesterol")
    plots["Probability vs Cholesterol"] = plot_to_img(fig)

    return render_template("visualizations.html", plots=plots, role="Admin")


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)





