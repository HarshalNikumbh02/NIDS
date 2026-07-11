from pymongo import MongoClient

# 1. Connect to your database (Make sure this matches your app.py connection string)
client = MongoClient("mongodb://localhost:27017/") 
db = client["your_database_name"] # Replace with your actual DB name
users_collection = db["users"]    # Replace with your actual collection name

# 2. Upgrade the specific user by their email
target_email = "your_email@example.com" # Put your registration email here

users_collection.update_one(
    {"email": target_email},
    {"$set": {"role": "admin"}}
)

print(f"Success! {target_email} is now an Admin.")