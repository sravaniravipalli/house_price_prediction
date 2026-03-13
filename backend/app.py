from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from pathlib import Path
import json
import os

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests for React frontend

# Load model and scaler
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
model = pickle.load(open(MODEL_DIR / "house_price_model.pkl", "rb"))
scaler = pickle.load(open(MODEL_DIR / "scaler.pkl", "rb"))

# Wishlist storage file
WISHLIST_FILE = BASE_DIR / "wishlist.json"

def load_wishlist():
    if WISHLIST_FILE.exists():
        with open(WISHLIST_FILE, "r") as f:
            return json.load(f)
    return {}

def save_wishlist(data):
    with open(WISHLIST_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ─── Predict Route ───────────────────────────────────────────────────────────

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({
            'message': 'Send a POST request with JSON: {"bedrooms":..., "bathrooms":..., "livingArea":..., "condition":..., "schoolsNearby":...}',
            'example': {
                'bedrooms': 3,
                'bathrooms': 2,
                'livingArea': 2000,
                'condition': 3,
                'schoolsNearby': 2
            }
        })
    data = request.json
    features = np.array([[data['bedrooms'], data['bathrooms'], data['livingArea'],
                          data['condition'], data['schoolsNearby']]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    return jsonify({'predicted_price': prediction * 9})


# ─── Properties Route ─────────────────────────────────────────────────────────

PROPERTIES_FILE = BASE_DIR / "properties.json"

def load_properties():
    if PROPERTIES_FILE.exists():
        with open(PROPERTIES_FILE, "r") as f:
            return json.load(f)
    return []

def save_properties(data):
    with open(PROPERTIES_FILE, "w") as f:
        json.dump(data, f, indent=2)

@app.route("/properties", methods=["GET"])
def get_properties():
    return jsonify({"properties": load_properties()})

@app.route("/properties", methods=["POST"])
def add_property():
    data = request.json
    if not data:
        return jsonify({"error": "Property data is required"}), 400
    properties = load_properties()
    properties.append(data)
    save_properties(properties)
    return jsonify({"message": "Property saved", "property": data}), 201


# ─── Wishlist Routes ──────────────────────────────────────────────────────────

# GET /wishlist/<userId> — get all wishlist items for a user
@app.route('/wishlist/<user_id>', methods=['GET'])
def get_wishlist(user_id):
    wishlist = load_wishlist()
    user_wishlist = wishlist.get(user_id, [])
    return jsonify({'wishlist': user_wishlist})


# POST /wishlist/<userId> — add a property to wishlist
@app.route('/wishlist/<user_id>', methods=['POST'])
def add_to_wishlist(user_id):
    data = request.json
    if not data or 'id' not in data:
        return jsonify({'error': 'Property data with id is required'}), 400

    wishlist = load_wishlist()
    if user_id not in wishlist:
        wishlist[user_id] = []

    # Avoid duplicates
    already_exists = any(item['id'] == data['id'] for item in wishlist[user_id])
    if already_exists:
        return jsonify({'message': 'Already in wishlist', 'wishlist': wishlist[user_id]}), 200

    wishlist[user_id].append(data)
    save_wishlist(wishlist)
    return jsonify({'message': 'Added to wishlist', 'wishlist': wishlist[user_id]}), 201


# DELETE /wishlist/<userId>/<propertyId> — remove a property from wishlist
@app.route('/wishlist/<user_id>/<property_id>', methods=['DELETE'])
def remove_from_wishlist(user_id, property_id):
    wishlist = load_wishlist()
    if user_id not in wishlist:
        return jsonify({'error': 'User not found'}), 404

    wishlist[user_id] = [item for item in wishlist[user_id] if str(item['id']) != str(property_id)]
    save_wishlist(wishlist)
    return jsonify({'message': 'Removed from wishlist', 'wishlist': wishlist[user_id]})


if __name__ == '__main__':
    app.run(debug=True)