import streamlit as st
import pandas as pd
import joblib # Or import pickle if you used that to save models
import os
import re

# --- Configuration ---
MODELS_DIR = 'prediction_models'
WORDCLOUDS_DIR = 'wordclouds_by_category'
SCORE_THRESHOLD = 200
# Define the expected features for the models IN THE CORRECT ORDER
# IMPORTANT: This order must match the order used when training the models!
EXPECTED_FEATURES = ['reply_count', 'thumbs_up', 'thumbs_down', 'stars']

# --- Helper Functions ---

def clean_category_name_for_files(category_name):
    """Cleans category name to match file/folder naming convention."""
    # This should match the cleaning logic used in recipe_cloud_generator.py
    # Example: Replace spaces and '&' with underscores, remove other special chars
    name = category_name.replace(' & ', '_') # Handle specific replacements first
    name = re.sub(r'[^\w\-_\.]', '_', name) # Replace remaining invalid chars with underscore
    return name

@st.cache_resource # Cache the loaded model to improve performance
def load_model(category_name):
    """Loads the pre-trained model for the selected category."""
    model_filename = f"{clean_category_name_for_files(category_name)}_model.pkl" # Assuming this naming convention
    model_path = os.path.join(MODELS_DIR, model_filename)
    try:
        model = joblib.load(model_path) # Or pickle.load(open(model_path, 'rb'))
        print(f"Successfully loaded model from {model_path}")
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found for category '{category_name}' at {model_path}. Please ensure the model exists and the naming is correct.")
        return None
    except Exception as e:
        st.error(f"Error loading model for '{category_name}': {e}")
        return None

def get_wordcloud_paths(category_name):
    """Gets the paths for the low and high score word clouds."""
    safe_category_name = clean_category_name_for_files(category_name)
    category_folder = os.path.join(WORDCLOUDS_DIR, safe_category_name)

    low_score_img = os.path.join(category_folder, f"low_score_{safe_category_name}.png")
    high_score_img = os.path.join(category_folder, f"high_score_{safe_category_name}.png")

    # Check if files exist
    if not os.path.exists(low_score_img):
        low_score_img = None
        st.warning(f"Low score word cloud not found for {category_name} at {low_score_img}")
    if not os.path.exists(high_score_img):
       high_score_img = None
       st.warning(f"High score word cloud not found for {category_name} at {high_score_img}")

    return low_score_img, high_score_img

# --- Define Categories (Manually or derive from folder names) ---
# Option 1: Manually define (ensure names match your folders/models after cleaning)
CATEGORIES = [
    'Soups & Chilis', 'Breads & Muffins', 'Breakfast', 'Cakes & Cupcakes',
    'Main Dishes', 'Pies & Tarts', 'Casseroles & Bakes', 'Pasta & Lasagna',
    'Salads & Sides', 'Dessert Specialties', 'Bars & Cookies'
]
# # Option 2: Derive from word cloud folder names (more robust if folders are correct)
# try:
#     raw_folder_names = [d for d in os.listdir(WORDCLOUDS_DIR) if os.path.isdir(os.path.join(WORDCLOUDS_DIR, d))]
#     # Attempt to convert folder names back to original category names (might need adjustment)
#     CATEGORIES = sorted([name.replace('_', ' & ').replace('_', ' ') for name in raw_folder_names]) # Simple replacement, adjust if needed
#     if not CATEGORIES:
#         st.error(f"Could not find any category folders in '{WORDCLOUDS_DIR}'. Please check the directory.")
#         CATEGORIES = ["Error - Check Folders"] # Fallback
# except FileNotFoundError:
#      st.error(f"Word cloud directory '{WORDCLOUDS_DIR}' not found.")
#      CATEGORIES = ["Error - Directory Missing"] # Fallback
# except Exception as e:
#      st.error(f"Error reading category folders: {e}")
#      CATEGORIES = ["Error - Reading Folders"] # Fallback


# --- Streamlit App Layout ---
st.set_page_config(layout="wide") # Use wider layout

st.title("🍲 Recipe Score Predictor & Analysis")
st.markdown("Select a recipe category, input user interaction metrics, and predict the 'Best Score'. See word clouds representing common terms in low-scoring and high-scoring recipe reviews.")

# --- Sidebar for Inputs ---
st.sidebar.header("Prediction Inputs")

# Category Selection
selected_category = st.sidebar.selectbox(
    "Select Recipe Category:",
    options=CATEGORIES,
    index=0 # Default to the first category
)

# Input Parameters
st.sidebar.markdown("---") # Separator
reply_count = st.sidebar.number_input("Number of Replies", min_value=0, value=0, step=1)
thumbs_up = st.sidebar.number_input("Thumbs Up Count 👍", min_value=0, value=5, step=1)
thumbs_down = st.sidebar.number_input("Thumbs Down Count 👎", min_value=0, value=0, step=1)
stars = st.sidebar.slider("Star Rating ⭐", min_value=0, max_value=5, value=4, step=1) # Assuming stars are 0-5

# --- Main Area for Results ---

if selected_category and "Error" not in selected_category: # Check if a valid category is selected
    st.header(f"Analysis for: {selected_category}")

    # --- Prediction Section ---
    st.subheader("📈 Score Prediction")

    # Load the model for the selected category
    model = load_model(selected_category)

    if model:
        # Prepare input data for prediction
        # IMPORTANT: Create a DataFrame with columns in the SAME order as training data
        input_data = pd.DataFrame([[reply_count, thumbs_up, thumbs_down, stars]],
                                   columns=EXPECTED_FEATURES)

        st.write("Input Features:")
        st.dataframe(input_data)

        # Make prediction
        try:
            predicted_score = model.predict(input_data)[0]
            # Display prediction
            st.metric(label="Predicted Best Score", value=f"{predicted_score:.2f}") # Format to 2 decimal places

            score_comparison = "Above" if predicted_score >= SCORE_THRESHOLD else "Below"
            st.info(f"The predicted score is **{score_comparison}** the threshold of {SCORE_THRESHOLD}.")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.error("Please ensure the input values are valid and the model was trained with these features in the correct order.")

    # --- Word Cloud Section ---
    st.subheader("💬 Word Clouds")
    st.markdown(f"Common words in reviews for **{selected_category}** recipes.")

    low_score_img_path, high_score_img_path = get_wordcloud_paths(selected_category)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Reviews with Score < {SCORE_THRESHOLD}**")
        if low_score_img_path:
            st.image(low_score_img_path, use_column_width=True)
        else:
            st.warning("Low score word cloud image not found.")

    with col2:
        st.markdown(f"**Reviews with Score >= {SCORE_THRESHOLD}**")
        if high_score_img_path:
            st.image(high_score_img_path, use_column_width=True)
        else:
            st.warning("High score word cloud image not found.")

else:
    st.warning("Please select a valid recipe category from the sidebar.")

st.sidebar.markdown("---")
st.sidebar.info("App developed based on recipe review data.")