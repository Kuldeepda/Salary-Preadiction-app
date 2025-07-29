import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

# Define job titles consistently.
# THIS LIST MUST CONTAIN ALL UNIQUE JOB TITLES (INCLUDING 'OTHERS')
# THAT YOUR MODEL WAS TRAINED ON FOR ONE-HOT ENCODING.
# If your model expects 55 job title features, this list should ideally have 55 entries.
JOB_TITLES = ['Content Marketing Manager', 'Data Analyst', 'Data Scientist', 'Digital Marketing Manager', 'Digital Marketing Specialist', 'Director of Data Science', 'Director of HR', 'Director of Marketing', 'Director of Operations', 'Financial Analyst', 'Financial Manager', 'Front End Developer', 'Front end Developer', 'Full Stack Engineer', 'Graphic Designer', 'Human Resources Coordinator', 'Human Resources Manager', 'Junior Data Analyst', 'Junior HR Coordinator', 'Junior HR Generalist', 'Junior Marketing Manager', 'Junior Sales Associate', 'Junior Sales Representative', 'Junior Software Developer', 'Junior Software Engineer', 'Junior Web Developer', 'Marketing Analyst', 'Marketing Coordinator', 'Marketing Director', 'Marketing Manager', 'Operations Manager', 'Others', 'Product Designer', 'Product Manager', 'Project Manager', 'Receptionist', 'Research Director', 'Research Scientist', 'Sales Associate', 'Sales Director', 'Sales Executive', 'Sales Manager', 'Sales Representative', 'Senior Data Scientist', 'Senior HR Generalist', 'Senior Human Resources Manager', 'Senior Product Marketing Manager', 'Senior Project Engineer', 'Senior Research Scientist', 'Senior Software Engineer', 'Social Media Manager', 'Software Developer', 'Software Engineer', 'Software Engineer Manager', 'Web Developer']




# Page configuration
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Simple Salary Prediction App")
st.write("Enter the details to get a salary prediction.")

@st.cache_resource
def load_model():
    """Load the trained model with comprehensive error handling"""
    try:
        model = joblib.load('salary_prediction_model.pkl')
        if not hasattr(model, 'predict'):
            st.error("❌ Loaded object is not a valid prediction model!")
            return None
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'salary_prediction_model.pkl' not found!")
        st.markdown("""
        <div style="background-color: #f8d7da; padding: 1rem; border-radius: 8px; border-left: 4px solid #dc3545; margin: 1rem 0;">
        <strong>Solution:</strong><br>
        • Ensure the model file is in the same directory as this script<br>
        • Check the file name is exactly 'salary_prediction_model.pkl'<br>
        • Make sure you have trained and saved the model first
        </div>
        """, unsafe_allow_html=True)
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.markdown("""
        <div style="background-color: #f8d7da; padding: 1rem; border-radius: 8px; border-left: 4px solid #dc3545; margin: 1rem 0;">
        <strong>Possible Issues:</strong><br>
        • Model file is corrupted<br>
        • Version mismatch between joblib/scikit-learn<br>
        • Model was saved with different Python version
        </div>
        """, unsafe_allow_html=True)
        return None

def get_model_info(model):
    """Get information about the loaded model"""
    info = {}
    try:
        if hasattr(model, 'n_features_in_'):
            info['n_features'] = model.n_features_in_
        elif hasattr(model, 'coef_') and hasattr(model, 'feature_names_in_'): # For models like LinearRegression
            info['n_features'] = len(model.coef_) if model.coef_.ndim == 1 else model.coef_.shape[1]
        elif hasattr(model, 'feature_importances_'): # For tree-based models
             info['n_features'] = len(model.feature_importances_)
        else:
            info['n_features'] = None

        info['model_type'] = type(model).__name__
    except Exception as e:
        st.warning(f"Could not extract full model info: {e}")
    return info

def create_feature_vector(age, gender, education, experience, job_title, model_info, all_job_titles_trained):
    """
    Create feature vector for prediction, adjusting for expected feature count.
    'all_job_titles_trained' should be the exhaustive list of job titles the model saw during training.
    """
    basic_features = [age, gender, education, experience]

    if model_info and model_info.get('n_features'):
        expected_total_features = model_info['n_features']
        # Assuming first 4 features are basic numerical ones
        expected_job_title_features = expected_total_features - 4

        if expected_job_title_features > 0:
            # Create a base for all possible job title dummy features, initialized to 0
            job_dummies_dict = {title: 0 for title in all_job_titles_trained}

            # Set the dummy for the selected job title to 1
            if job_title in job_dummies_dict:
                job_dummies_dict[job_title] = 1
            elif job_title == 'Others' and 'Others' in job_dummies_dict:
                 job_dummies_dict['Others'] = 1
            else:
                # Fallback if job_title is not directly found in the exhaustive list
                # This might happen if 'Others' isn't explicitly defined or the list is incomplete.
                # For safety, if 'Others' is a trained category, set it to 1 if the input job isn't found.
                if 'Others' in job_dummies_dict:
                    job_dummies_dict['Others'] = 1 # Assign to 'Others' category if specific job not found

            # Convert the dictionary values to a list in a consistent order
            # The order must match the order of dummy columns used during training.
            # This is crucial! Using `all_job_titles_trained` as the order reference.
            job_dummies = [job_dummies_dict[title] for title in all_job_titles_trained]

            features = basic_features + job_dummies
            return np.array(features).reshape(1, -1)
        else:
            # Model expects only basic features, no job title dummies
            return np.array(basic_features).reshape(1, -1)
    else:
        # Fallback if model info isn't available or n_features is not set
        st.warning("Could not determine expected feature count from model. Attempting basic prediction.")
        return np.array(basic_features).reshape(1, -1)


# Load the model
model = load_model()

if model is None:
    st.stop()

# Get model information (crucial for feature count)
model_info = get_model_info(model)
if model_info.get('n_features') is None:
    st.error("❌ Unable to determine the number of features expected by the model. Cannot proceed with prediction.")
    st.stop()


# Input fields
age = st.slider("Age", min_value=18, max_value=65, value=30, step=1)
years_experience = st.slider("Years of Experience", min_value=0, max_value=40, value=5, step=1)

gender_options = {"Male": 0, "Female": 1, "Other": 2}
gender = st.selectbox("Gender", options=list(gender_options.keys()))
gender_encoded = gender_options[gender]

education_options = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
education_level = st.selectbox("Education Level", options=list(education_options.keys()))
education_encoded = education_options[education_level]

job_title = st.selectbox("Job Title", options=JOB_TITLES) # Use the defined JOB_TITLES list

if st.button("Predict Salary"):
    # Ensure JOB_TITLES list is accurate for the model's training
    expected_job_title_features = model_info['n_features'] - 4 # Assuming 4 numerical features
    if len(JOB_TITLES) != expected_job_title_features:
        st.warning(f"""
        ⚠️ **Feature Mismatch Warning:** The model expects {expected_job_title_features} one-hot encoded job title features,
        but your `JOB_TITLES` list in `app.py` has {len(JOB_TITLES)} entries.
        For accurate prediction, this list should precisely match all unique job titles
        your model was trained on after any grouping.
        """)
        # If the number of job titles doesn't match, we need to try to infer or pad.
        # This is where the assumption/heuristic comes in.
        # For a truly robust solution, you'd save the OHE column names during training.
        # For now, we will use the current JOB_TITLES as the "all_job_titles_trained" reference
        # and let `create_feature_vector` pad/handle.
        all_job_titles_for_prediction = JOB_TITLES
        if len(JOB_TITLES) < expected_job_title_features:
            # Pad with generic names if JOB_TITLES is too short, to match expected count
            # This is a dangerous assumption if the order isn't correct.
            # The ideal solution is for the user to provide the correct, exhaustive JOB_TITLES.
            padded_titles = [f"Unknown_Job_{i}" for i in range(expected_job_title_features - len(JOB_TITLES))]
            all_job_titles_for_prediction = JOB_TITLES + padded_titles
            st.warning("Attempting to pad job titles to match expected feature count. Ensure your `JOB_TITLES` list is complete for accuracy.")
        elif len(JOB_TITLES) > expected_job_title_features:
            # Trim if JOB_TITLES is too long
            all_job_titles_for_prediction = JOB_TITLES[:expected_job_title_features]
            st.warning("Your `JOB_TITLES` list is longer than the model expects for job features. Trimming the list.")

    else:
        all_job_titles_for_prediction = JOB_TITLES

    # Create feature vector
    features = create_feature_vector(
        age, gender_encoded, education_encoded, years_experience, job_title,
        model_info, all_job_titles_for_prediction
    )

    if features.shape[1] != model_info['n_features']:
        st.error(f"❌ Internal feature creation mismatch: Expected {model_info['n_features']} features, but created {features.shape[1]}.")
        st.stop()

    try:
        prediction = model.predict(features)[0]
        st.success(f"Predicted Monthly Salary: **Rs. {prediction:,.2f}**")
        st.info("Please note: This is an estimated salary based on the trained model. Actual salaries may vary.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure your input features exactly match what the model was trained on, especially the job title categories and their order in one-hot encoding.")

st.markdown("---")
st.markdown("Developed by Kuldeep Das")