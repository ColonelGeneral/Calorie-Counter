import os
import json
import pandas as pd
import numpy as np
import requests
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from tensorflow import keras
from dotenv import load_dotenv
import time
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, date, timedelta

# ========================== Environment & Config ==========================
load_dotenv()
USDA_API_KEY = os.getenv("USDA_API_KEY")

# Flask app configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SECRET_KEY'] = 'your-secret-key'

app.config['secret-key'] = "6bbbfc70cd6ece14a9adb319073ca49dfa684dd7f62231a806302496d8a59ac3"
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///meals.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Globals
model = None
class_indices = None


# ========================== Helper Functions ==========================
def allowed_file(filename):
    """Check allowed file types"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model_and_data():
    """Load model and corrected class indices"""
    global model, class_indices
    try:
        model_path = 'fine_tuned_food_model.h5'
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            print("✅ Fine-tuned model loaded successfully!")
        else:
            print(f"❌ Model file {model_path} not found!")

        if os.path.exists("class_indices.json"):
            with open("class_indices.json", "r") as f:
                class_indices_data = json.load(f)
            # Convert {food_name: index} → {index: food_name}
            class_indices = {int(v): k for k, v in class_indices_data.items()}
            print(f"✅ Class indices loaded successfully! Total classes: {len(class_indices)}")
        else:
            print("⚠️ class_indices.json not found — predictions may be unnamed.")
            class_indices = {}

    except Exception as e:
        print(f"❌ Error loading model/data: {e}")


def preprocess_image(img_path, target_size=(224, 224)):
    """Prepare image for prediction"""
    try:
        img = keras.utils.load_img(img_path, target_size=target_size)
        img_array = keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array / 255.0
    except Exception as e:
        print(f"⚠️ Error preprocessing image: {e}")
        return None


def predict_food(image_path):
    """Predict food from uploaded image"""
    if model is None:
        print("⚠️ Model not loaded.")
        return None, 0.0
    try:
        img_array = preprocess_image(image_path)
        predictions = model.predict(img_array, verbose=0)
        idx = np.argmax(predictions[0])
        confidence = float(predictions[0][idx])
        food_name = class_indices.get(idx, "Unknown")

        # Beautify label
        if food_name != "Unknown":
            food_name = food_name.replace("_", " ").title()

        print(f"🍽 Predicted: {food_name} ({confidence:.2f})")
        return food_name, confidence

    except Exception as e:
        print(f"❌ Error predicting food: {e}")
        return None, 0.0


# ========================== Nutrition Info (USDA API + CSV fallback) ==========================
def get_nutrition_info(food_name, portion_size=100):
    """Fetch nutrition info using USDA FoodData Central API"""
    try:
        # 1️⃣ Local CSV fallback
        csv_path = "nutrition_data.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            match = df[df["Food"].str.lower() == food_name.lower()]
            if not match.empty:
                print(f"📦 Found '{food_name}' in local CSV data")
                base = {
                    "Calories": float(match["Calories"].iloc[0]),
                    "Protein": float(match["Protein"].iloc[0]),
                    "Carbs": float(match["Carbs"].iloc[0]),
                    "Fat": float(match["Fat"].iloc[0]),
                }
                factor = portion_size / 100.0
                total = {k: round(v * factor, 1) for k, v in base.items()}
                return {
                    "per_100g": base,
                    "total": total,
                    "portion_size": portion_size,
                    "source": "local"
                }

        # 2️⃣ Fetch from USDA API
        if not USDA_API_KEY:
            print("⚠️ USDA_API_KEY missing in .env")
            return None

        print(f"🌐 Fetching '{food_name}' from USDA API...")
        search_url = f"https://api.nal.usda.gov/fdc/v1/foods/search"
        params = {
            "query": food_name,
            "pageSize": 1,
            "api_key": USDA_API_KEY
        }

        res = requests.get(search_url, params=params, timeout=10)
        if res.status_code != 200:
            print(f"⚠️ USDA API request failed ({res.status_code})")
            return None

        data = res.json()
        if "foods" not in data or not data["foods"]:
            print(f"⚠️ No USDA data found for '{food_name}'")
            return None

        nutrients = data["foods"][0].get("foodNutrients", [])
        nutrition = {"Calories": 0, "Protein": 0, "Carbs": 0, "Fat": 0}

        for n in nutrients:
            name = n.get("nutrientName", "").lower()
            if "energy" in name or "calorie" in name:
                nutrition["Calories"] = round(n.get("value", 0), 1)
            elif "protein" in name:
                nutrition["Protein"] = round(n.get("value", 0), 1)
            elif "carbohydrate" in name:
                nutrition["Carbs"] = round(n.get("value", 0), 1)
            elif "fat" in name:
                nutrition["Fat"] = round(n.get("value", 0), 1)

        factor = portion_size / 100.0
        total = {k: round(v * factor, 1) for k, v in nutrition.items()}

        print(f"✅ USDA data fetched successfully for '{food_name}'")
        return {
            "per_100g": nutrition,
            "total": total,
            "portion_size": portion_size,
            "source": "usda"
        }

    except Exception as e:
        print(f"❌ Error fetching USDA data: {e}")
        return None

class MealLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    food_name = db.Column(db.String(120), nullable=False)
    calories = db.Column(db.Float, nullable=False)
    protein = db.Column(db.Float)
    carbs = db.Column(db.Float)
    fat = db.Column(db.Float)
    portion_size = db.Column(db.Integer)
    logged_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Meal {self.food_name} ({self.calories} kcal)>"

# ========================== Flask Routes ==========================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        portion_size = min(max(request.form.get('portion_size', 100, type=int), 50), 500)

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = f"{int(time.time())}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        food_name, confidence = predict_food(filepath)
        nutrition_info = get_nutrition_info(food_name, portion_size)

        result_data = {
            'image_path': url_for('static', filename=f'uploads/{filename}'),
            'food_name': food_name,
            'confidence': round(confidence * 100, 2),
            'nutrition_info': nutrition_info,
            'portion_size': portion_size
        }

        return render_template(
    'result.html',
    **result_data,
    calories=result_data["nutrition_info"]["total"]["Calories"],
    protein=result_data["nutrition_info"]["total"]["Protein"],
    carbs=result_data["nutrition_info"]["total"]["Carbs"],
    fat=result_data["nutrition_info"]["total"]["Fat"]
)

    except Exception as e:
        print(f"Error in /predict route: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/weekly_summary')
def weekly_summary():
    try:
        # Determine date range: today and 6 days earlier
        today = date.today()
        start_date = today - timedelta(days=6)

        # Query all meals from last 7 days
        meals = MealLog.query.filter(MealLog.logged_at >= datetime(start_date.year, start_date.month, start_date.day)).all()

        # Prepare a dictionary keyed by date
        daily_totals = {}
        for i in range(7):
            d = start_date + timedelta(days=i)
            daily_totals[d] = 0  # initialize

        # Sum calories by day
        for meal in meals:
            meal_day = meal.logged_at.date()
            if meal_day in daily_totals:
                daily_totals[meal_day] += meal.calories

        # Convert to template-friendly list
        weekly_data = []
        for d, total in daily_totals.items():
            weekly_data.append({
                "date": d.strftime("%Y-%m-%d"),
                "calories": round(total, 1)
            })

        # Weekly total
        weekly_total = round(sum(item["calories"] for item in weekly_data), 1)

        return render_template(
            "weekly_summary.html",
            weekly_data=weekly_data,
            weekly_total=weekly_total
        )

    except Exception as e:
        print("❌ Error in /weekly_summary:", e)
        return f"Error: {e}", 500

    
@app.route('/log_meal', methods=['POST'])
def log_meal():
    try:
        food_name = request.form['food_name']
        calories = float(request.form['calories'])
        protein = float(request.form.get('protein', 0))
        carbs = float(request.form.get('carbs', 0))
        fat = float(request.form.get('fat', 0))
        portion_size = int(request.form.get('portion_size', 100))

        meal = MealLog(
            food_name=food_name,
            calories=calories,
            protein=protein,
            carbs=carbs,
            fat=fat,
            portion_size=portion_size
        )
        db.session.add(meal)
        db.session.commit()

        return render_template("log_success.html", food_name=food_name, calories=calories)

    except Exception as e:
        return f"Error logging meal: {e}", 500


@app.route('/history')
def history():
    meals = MealLog.query.order_by(MealLog.logged_at.desc()).all()
    return render_template("history.html", meals=meals)


@app.route('/daily_summary')
def daily_summary():
    today = date.today()
    start = datetime(today.year, today.month, today.day)

    meals = MealLog.query.filter(MealLog.logged_at >= start).all()
    total_calories = sum(m.calories for m in meals)

    return render_template("summary.html",
                           total=total_calories,
                           count=len(meals),
                           meals=meals)



# ========================== Main ==========================
if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    load_model_and_data()
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)
