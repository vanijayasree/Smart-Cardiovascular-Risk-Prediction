# Cardiovascular Risk Prediction System

A Flask web application for predicting cardiovascular disease risk using machine learning,
with personalized recommendations for diet, exercise, and stress management.

**Database:** MySQL (as documented in the original project design)

---

## Quick Start

**Windows Users:** See [SETUP_WINDOWS.md](SETUP_WINDOWS.md) for detailed setup instructions.

### Prerequisites
- Python 3.8+
- MySQL Server 8.0 (running on localhost)
- pip (Python package manager)

### Installation
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up admin user (run once)
python setup.py

# 3. Run the application
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## Default Credentials

| Role | Username | Password |

Register your own user account from the Register page to test as a regular user.

---

## Features

### Core Prediction
- **Input:** Patient health parameters (age, cholesterol, blood pressure, ECG, etc.)
- **Output:** Risk level classification (Low / Borderline / High)
- **Probability:** Confidence score of the prediction

### Personalized Recommendations
- **Diet Plan** - Foods and eating habits based on risk level
- **Exercise Plan** - Physical activity recommendations
- **Stress Management** - Coping strategies and relaxation techniques

### Reports & Data
- **PDF Reports** - Download branded reports with patient data and recommendations
- **Data Visualization** - Risk distribution, age vs risk, cholesterol trends
- **Model Evaluation** - Live comparison of Logistic Regression, Decision Tree, and Random Forest
- **Export** - CSV export of all prediction records

### Administration
- **User Management** - View registered users, manage roles
- **Report Management** - View/delete all predictions
- **System Settings** - Maintenance mode, registration control, theme selection
- **Analytics** - System-wide visualizations and statistics

---

## Database Schema

**Host:** localhost  
**User:** root  
**Password:** *******  
**Database:** heart_project

### Tables
- **users** - User accounts with roles (Admin/User)
- **heart_reports** - All prediction records with health data, risk level, recommendations
- **system_settings** - System configuration and settings

---

## Project Structure

```
├── app.py                   # Flask application with all routes
├── setup.py                 # Admin user setup script
├── requirements.txt         # Python dependencies
├── training.py              # ML model training (reference)
├── heart_clean.csv          # Dataset for evaluation page
│
├── models/
│   ├── heart_model.pkl      # Trained ML model (scikit-learn)
│   └── scaler.pkl           # Feature scaler
│
├── templates/               # HTML pages (Bootstrap 5 UI)
│   ├── base.html            # Base template with navbar
│   ├── index.html           # Home page
│   ├── login.html           # Login page
│   ├── register.html        # Registration page
│   ├── user_dashboard.html  # User predictions
│   ├── user_predict.html    # Prediction form
│   ├── admin_dashboard.html # Admin panel
│   ├── user_dashboard.html  # User reports history
│   └── ... (more pages)
│
└── static/
    ├── css/                 # Custom stylesheets
    ├── js/                  # JavaScript utilities
    └── images/              # Logos, backgrounds, icons
```

---

## ML Model Details

**Algorithm:** Ensemble of multiple classifiers
- Logistic Regression
- Decision Tree
- Random Forest

**Features:** 13 clinical parameters
- Age, Sex, Chest Pain Type
- Resting Blood Pressure, Cholesterol
- Fasting Blood Sugar, Resting ECG
- Max Heart Rate, Exercise-Induced Angina
- ST Depression, ST Slope
- And more...

**Accuracy:** ~85-90% on test data (see Model Evaluation page in app)

---

## Troubleshooting

**MySQL connection error?**
- Ensure MySQL80 service is running (`Win + R` → `services.msc`)
- Verify password: `********`
- Check database `heart_project` exists

**Port 5000 already in use?**
- Change port in `app.py`: `app.run(debug=True, port=5001)`

**Import errors?**
- Reinstall dependencies: `pip install --upgrade -r requirements.txt`

---

## Security Notes

- Passwords are hashed with werkzeug.security.generate_password_hash
- Change `app.secret_key` before production deployment
- Default admin account is for demonstration only
- SQL queries use parameterized statements to prevent injection

---

## Technologies

- **Backend:** Flask (Python web framework)
- **Database:** MySQL
- **ML:** scikit-learn (Logistic Regression, Decision Tree, Random Forest)
- **Frontend:** Bootstrap 5, HTML5, CSS3, JavaScript
- **Visualization:** Matplotlib, Seaborn
- **Reports:** ReportLab (PDF generation)

---

## Submission Details

This project implements:
✓ User authentication system (login/register)
✓ Role-based access control (Admin/User)
✓ Machine learning prediction model
✓ Database integration (MySQL)
✓ Professional UI with Bootstrap 5
✓ PDF report generation
✓ Data visualization
✓ Admin dashboard

---

## License

Educational project - use for learning purposes.

---

**For detailed Windows setup instructions, see [SETUP_WINDOWS.md](SETUP_WINDOWS.md)**
