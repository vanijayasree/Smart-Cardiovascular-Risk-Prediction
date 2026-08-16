"""
Setup script to create or reset the default admin user in MySQL.
Run this whenever you want to ensure the admin account exists
with the default password.
"""

import mysql.connector
from werkzeug.security import generate_password_hash
import os
from dotenv import load_dotenv
load_dotenv()

def setup_admin():
    try:
        # Connect to MySQL
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password=os.getenv("DB_PASSWORD"),
            database="heart_project"
        )
        cursor = conn.cursor()

        # Default admin credentials
        username = os.getenv("ADMIN_USERNAME", "admin")
        password = os.getenv("ADMIN_PASSWORD")
        if not password:
            raise ValueError("ADMIN_PASSWORD is not set in the .env file")

        hashed_password = generate_password_hash(password)

        # Check if admin already exists
        cursor.execute("SELECT COUNT(*) FROM users WHERE username=%s", (username,))
        count = cursor.fetchone()[0]

        if count == 0:
            # Create new admin
            cursor.execute("""
                INSERT INTO users (username, password_hash, role, is_active)
                VALUES (%s, %s, %s, %s)
            """, (username, hashed_password, "Admin", 1))

            print("✓ Admin user created successfully!")

        else:
            # Reset existing admin password
            cursor.execute("""
                UPDATE users
                SET password_hash=%s,
                    role='Admin',
                    is_active=1
                WHERE username=%s
            """, (hashed_password, username))

            print("✓ Admin password reset successfully!")

        conn.commit()

        print("--------------------------------")
        print(f"Username : {username}")
        print("--------------------------------")

        cursor.close()
        conn.close()

    except mysql.connector.Error as err:
        print(f"✗ Database Error: {err}")
        print("\nCheck the following:")
        print("1. MySQL Server is running.")
        print("2. Database 'heart_project' exists.")
        print("3. MySQL password is correct.")

if __name__ == "__main__":
    setup_admin()