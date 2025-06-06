#!/usr/bin/env python3
"""
Script to view users and add test users with face embeddings
"""

import sys
import os
import sqlite3
import json
import requests
import base64
from pathlib import Path

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def check_users():
    """Check existing users in the database"""
    db_path = Path(__file__).parent / "facesocial.db"
    
    if not db_path.exists():
        print("Database does not exist!")
        return
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Check if users table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
    if not cursor.fetchone():
        print("Users table does not exist!")
        conn.close()
        return
    
    # Get all users
    cursor.execute("SELECT id, username, email, full_name, face_embedding FROM users")
    users = cursor.fetchall()
    
    print(f"Found {len(users)} users:")
    for user in users:
        user_id, username, email, full_name, face_embedding = user
        has_embedding = "Yes" if face_embedding else "No"
        print(f"  ID: {user_id}, Username: {username}, Email: {email}, Name: {full_name}, Face Embedding: {has_embedding}")
    
    conn.close()
    return users

def add_test_user_with_face():
    """Add a test user with face embedding"""
    print("\nAdding test user with face embedding...")
      # First, extract face embedding from test image
    test_image_path = Path(__file__).parent.parent / "test_images" / "group_test_01.jpg"
    
    if not test_image_path.exists():
        print(f"Test image not found: {test_image_path}")
        return
    
    try:
        # Read and encode image
        with open(test_image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Extract face embedding from the test image
        response = requests.post(
            "http://localhost:8000/api/v1/ai/extract-face-embedding",
            files={'image': open(test_image_path, 'rb')}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success') and result.get('embedding'):
                face_embedding = result['embedding']
                print(f"Successfully extracted face embedding with {len(face_embedding)} dimensions")
                
                # Add user to database
                db_path = Path(__file__).parent / "facesocial.db"
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                
                # Check if test user already exists
                cursor.execute("SELECT id FROM users WHERE username = ?", ("testuser",))
                if cursor.fetchone():
                    print("Test user already exists!")
                    conn.close()
                    return
                
                # Insert test user with face embedding
                cursor.execute("""
                    INSERT INTO users (username, email, hashed_password, full_name, face_embedding, is_active, is_verified)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    "testuser",
                    "test@example.com", 
                    "$2b$12$dummy_hash_for_testing",  # Dummy password hash
                    "Test User",
                    json.dumps(face_embedding),  # Store as JSON string
                    True,
                    True
                ))
                
                conn.commit()
                user_id = cursor.lastrowid
                print(f"Added test user with ID: {user_id}")
                conn.close()
                
            else:
                print("Failed to extract face embedding:", result.get('message'))
        else:
            print(f"Face embedding extraction failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Error adding test user: {e}")

if __name__ == "__main__":
    users = check_users()
    
    # If no users with face embeddings, add test user
    if not users or not any(user[4] for user in users):  # Check if any user has face_embedding
        add_test_user_with_face()
        print("\nUpdated user list:")
        check_users()
    else:
        print("\nUsers with face embeddings already exist.")
