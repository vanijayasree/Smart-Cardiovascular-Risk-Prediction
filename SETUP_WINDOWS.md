# Cardiovascular Risk Prediction System - Windows Setup Guide

## Prerequisites
✓ MySQL Server 8.0 installed and running (MySQL80 service)
✓ Python 3.8 or higher installed
✓ Git (optional, for version control)

---

## Step 1: Verify MySQL is Running

1. **Press `Win + R`**, type `services.msc` and press Enter
2. Look for **MySQL80** service
3. Make sure it says **"Running"** on the right
4. If it says "Stopped", right-click it and select "Start"

---

## Step 2: Extract and Setup Project

1. **Extract the project folder** to a location like:
   ```
   C:\Users\YourName\CardioVascular_Risk_Prediction
   ```

2. **Open Command Prompt** and navigate to the project:
   ```
   cd C:\Users\YourName\CardioVascular_Risk_Prediction
   ```

3. **Install Python dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Create the admin user** (run this once):
   ```
   python setup.py
   ```
   You should see:
   ```
   ✓ Admin user created successfully!
     Username: admin
     Password: admin123
   ```

---

## Step 3: Run the Application

1. **Start the Flask app:**
   ```
   python app.py
   ```

   You should see:
   ```
   Running on http://127.0.0.1:5000
   ```

2. **Open your browser** and go to:
   ```
   http://127.0.0.1:5000
   ```

3. **Login with:**
   - Username: `admin`
   - Password: `admin123`

---

## Features to Try

### As Admin:
- View all user predictions on the Admin Dashboard
- Manage users (see registered accounts)
- Export reports to CSV
- View system-wide visualizations

### As a Regular User:
1. **Register** a new account
2. **Login** with your credentials
3. **Make a Prediction** by filling in patient health data
4. **View Results** with risk level (Low / Borderline / High)
5. **Download PDF Report** with personalized recommendations
6. **View Recommendations** for diet, exercise, and stress management
7. **See Model Evaluation** (compares ML algorithms)
8. **View Visualizations** of risk distribution and statistics

---

## Database Details

**Database Name:** `heart_project`
**Host:** `localhost`
**User:** `root`
**Password:** `[REDACTED]`

Tables:
- `users` - User accounts with login credentials and roles
- `heart_reports` - All prediction records with health data and results
- `system_settings` - System configuration

---

## Troubleshooting

### Error: "Can't connect to MySQL server on 'localhost:3306'"
- **Solution:** Make sure MySQL80 service is running (see Step 1)
- If it's stopped, start it from Services

### Error: "Access denied for user 'root'"
- **Solution:** Check your password is `[REDACTED]` in the connection string
- Edit `app.py` line ~40 if your password is different

### Error: "No database selected"
- **Solution:** The database `heart_project` wasn't created properly
- Re-run the SQL script in MySQL Workbench to create it

### Port 5000 already in use
- **Solution:** Either:
  - Close the other program using port 5000
  - Or change Flask port in `app.py` line at the bottom: `app.run(debug=True, port=5001)`

---

## Project Structure
```
CardioVascular_Risk_Prediction/
├── app.py                    # Main Flask application
├── setup.py                  # Setup script (run once)
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview
├── heart_clean.csv          # Dataset for model evaluation
├── training.py              # ML model training script (reference)
├── models/
│   ├── heart_model.pkl      # Trained ML model
│   └── scaler.pkl           # Data scaler
├── templates/               # HTML pages (Bootstrap 5)
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── user_dashboard.html
│   ├── admin_dashboard.html
│   └── ... (more pages)
└── static/
    ├── css/                 # Stylesheets
    ├── js/                  # JavaScript
    └── images/              # Images and logos
```

---

## Notes for Submission

- Default admin credentials are baked in for easy testing
- Database credentials are in `app.py` (line ~40)
- Change `app.secret_key` before production use
- All user passwords are securely hashed with werkzeug
- SQLite files are NOT used; this uses MySQL as documented

---

## Support

If you have issues:
1. Check MySQL is running (MySQL80 service)
2. Verify the password: `[REDACTED]`
3. Make sure the database `heart_project` exists
4. Check Python 3.8+ is installed: `python --version`

Good luck with your project! 🚀
