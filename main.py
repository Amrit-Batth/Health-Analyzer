from flask import Flask, request, render_template, jsonify, redirect, session, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import os
import numpy as np
import pandas as pd
import pickle
from google_api_key import google_api_key
from routes.routes_medical import medical_bp
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from extensions import db
from models import User
import psycopg2


app = Flask(__name__)
app.secret_key = os.urandom(24)  # or use a proper static secret key for production
GOOGLE_API_KEY = google_api_key  # use this only for API requests

# ✅ PostgreSQL config
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:1234567890@localhost/flask-db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ✅ Init DB
db.init_app(app)

# ✅ Init Login Manager
login_manager = LoginManager()
login_manager.init_app(app)


# ✅ Create Tables
with app.app_context():
    db.create_all()
#<=================================================================================================================>


# ✅ Register Blueprints
app.register_blueprint(medical_bp)

# ✅ Load datasets
sym_des = pd.read_csv("./datasets/symtoms_df.csv")
precautions = pd.read_csv("./datasets/precautions_df.csv")
workout = pd.read_csv("./datasets/workout_df.csv")
description = pd.read_csv("./datasets/description.csv")
medications = pd.read_csv("./datasets/medications.csv")
diets = pd.read_csv("./datasets/diets.csv")

# ✅ Load model
svc = pickle.load(open("models/svc.pkl", "rb"))



# ========================== Helper Function ==========================
def helper(dis):
    desc_series = description[description['Disease'] == dis]['Description']
    desc = " ".join(desc_series.astype(str))

    pre_df = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = pre_df.values.tolist()

    med_series = medications[medications['Disease'] == dis]['Medication']
    med = med_series.tolist()

    diet_series = diets[diets['Disease'] == dis]['Diet']
    die = diet_series.tolist()

    workout_series = workout[workout['disease'] == dis]['workout']
    wrkout = workout_series.tolist()

    return desc, pre, med, die, wrkout


symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}


# ========================== Model Prediction Function ==========================
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    predicted_index = svc.predict([input_vector])[0]
    return diseases_list[predicted_index]


# ========================= Load User for Login Manager =========================
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ========================= Predict Route =========================
@app.route("/predict")
@login_required
def index():
    username = session.get('username')
    return render_template("index.html", username=username)


# ========================= Predict Route (POST & GET) =========================
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')

        # Check if input is empty
        if not symptoms:
            message = "Please enter symptoms. The input field cannot be empty."
            return render_template('index.html', message=message)

        symptoms = symptoms.strip()
        if symptoms.lower() == "symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms."
            return render_template('index.html', message=message)

        try:
            # Process and clean user input
            user_symptoms = [s.strip("[]' ") for s in symptoms.split(',')]
            predicted_disease = get_predicted_value(user_symptoms)

            # Fetch related data for the predicted disease
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)
            my_precautions = [i for i in precautions[0]]  # unpack nested list

            return render_template(
                'index.html',
                predicted_disease=predicted_disease,
                dis_des=dis_des,
                my_precautions=my_precautions,
                medications=medications,
                my_diet=rec_diet,
                workout=workout
            )
        except Exception as e:
            message = (
                "An error occurred: The provided symptoms are not available in our dataset. "
                "Please check your input and try again."
            )
            return render_template('index.html', message=message)

    return render_template('index.html')



# ======================== Static Pages Routes ========================

# About Page Route
@app.route('/about')
def about():
    return render_template("about.html")

# Symptoms Information Page Route
@app.route('/symptoms')
def symptoms():
    return render_template("symptoms.html")

# Blog Page Route
@app.route('/blog')
def blog():
    return render_template("blog.html")



# <======================================================================================================>

# Sign Up Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        # Validate input fields
        if not username or not password:
            flash("Username and password are required.", "warning")  # Adding category for flash
            return redirect('/signup')

        # Check if the username already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already taken. Try another one.", "warning")
            return redirect('/signup')

        # Hash the password and create a new user
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password)

        # Add and commit the new user to the database
        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully. Please sign in.", "success")
        return redirect('/')

    return render_template('signup.html')

# Sign In Route
@app.route('/', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        # Check if user exists and password is correct
        if user and check_password_hash(user.password, password):
            login_user(user)
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect('/predict')
        else:
            flash("Invalid username or password.", "danger")  # Added category for flash
            return redirect('/')

    return render_template('signin.html')


# Logout Route
@app.route('/logout')
@login_required
def logout():
    logout_user()          # Properly logs out the user from Flask-Login
    session.clear()        # Clear all session data
    flash("You have been logged out successfully.", "info")  # Optional: flash logout message
    return redirect('/')

# Get User Route
@app.route('/getuser')
@login_required
def get_user():
    return render_template('getuser.html', user=current_user)

  
# <========================================================================================================>

if __name__ == '__main__':
    # Initialize the database and create tables
    with app.app_context():
        db.create_all()
    app.run(debug=True)