"""
config.py

Central place for all configuration settings. Reads sensitive values
(passwords, secret keys) from the .env file instead of hardcoding them
directly in app.py.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file in the same folder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-fallback-key")

    # Database
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_USER = os.getenv("DB_USER", "root")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_NAME = os.getenv("DB_NAME", "heart_project")

    # File paths
    MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_model.pkl")
    SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
    DATASET_PATH = os.path.join(BASE_DIR, "heart_clean.csv")
    SHAP_DIR = os.path.join(BASE_DIR, "static", "shap_reports")
