# streamlit_app.py (or your filename)
import streamlit as st
import pandas as pd
import joblib
import os
import re
import sys # Keep for potential future debugging if needed

# --- Configuration ---
# Define directories for BOTH model sets
MODELS_DIR_FULL = 'prediction_models'
MODELS_DIR_NO_THUMBS = 'prediction_models_no_thumbs'
WORDCLOUDS_DIR = 'wordclouds_by_category'
SCORE_THRESHOLD = 200 # Example threshold

# Define EXPECTED features for BOTH model sets
# Order MUST exactly match the FEATURE_ORDER in the respective model generation scripts
EXPECTED_FEATURES_FULL = ['user_reputation', 'reply_count', 'thumbs_up', 'thumbs_down', 'stars']
EXPECTED_FEATURES_NO_THUMBS = ['user_reputation', 'reply_count', 'stars']

# --- Helper Functions ---

def clean_category_name_for_model_files(category_name):
    """Cleans category name to match the MODEL filename convention."""
    name = category_name.replace(' & ', '_and_')
    name = re.sub(r'[^\w\-\.]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name

def format_category_for_wordcloud_path(category_name):
    """Formats category name to match the WORD CLOUD folder/file convention."""
    name = category_name.replace(' & ', ' _ ')
    return name

# Use Streamlit's caching for models - Key depends on category AND mode
@st.cache_resource
def load_model(category_name, mode):
    """Loads the pre-trained model based on category and selected mode."""
    if mode == "With Thumbs Up/Down":
        model_dir = MODELS_DIR_FULL
        expected_features_len = len(EXPECTED_FEATURES_FULL)
    elif mode == "Without Thumbs Up/Down":
        model_dir = MODELS_DIR_NO_THUMBS
        expected_features_len = len(EXPECTED_FEATURES_NO_THUMBS)
    else:
        st.error(f"Invalid model mode selected: {mode}")
        return None

    model_formatted_name = clean_category_name_for_model_files(category_name)
    model_filename = f"{model_formatted_name}.pkl"
    model_path = os.path.join(model_dir, model_filename)

    print(f"[Model Load] Mode: '{mode}'. Attempting to load model from path: {model_path}")

    try:
        if not os.path.exists(model_path):
             st.error(f"Model file not found for mode '{mode}': '{model_path}'.")
             st.info(f"Ensure you have run the correct model generation script and the file exists in the '{model_dir}' folder.")
             print(f"[Error] Model file does not exist at expected path: {model_path}")
             return None

        model = joblib.load(model_path)
        print(f"[Model Load] Successfully loaded model for '{category_name}' (Mode: '{mode}').")

        # Check if model expects the correct number of features for the selected mode
        if hasattr(model, 'n_features_in_'):
             if model.n_features_in_ != expected_features_len:
                  st.error(f"Model Error (Mode: '{mode}'): Loaded model expects {model.n_features_in_} features, but app is configured for {expected_features_len} features in this mode.")
                  print(f"[Error] Feature mismatch: Model expects {model.n_features_in_}, App expects {expected_features_len} for mode '{mode}'")
                  return None
        else:
             print("[Warning] Loaded model does not have 'n_features_in_' attribute for verification.")

        return model

    except (AttributeError, ModuleNotFoundError, ImportError, EOFError, TypeError, ValueError) as load_err:
         st.error(f"Model Load Error for '{category_name}' (Mode: '{mode}'): {type(load_err).__name__} - {load_err}")
         st.error("Check for Python/library version mismatches or corrupt .pkl files.")
         st.warning(f"Ensure 'requirements.txt' matches the environment used for '{model_dir}'. Re-generating the .pkl files might be necessary.")
         print(f"[Error] Error loading {model_path}: {type(load_err).__name__} - {load_err}")
         return None
    except Exception as e:
        st.error(f"An unexpected error occurred loading model '{model_filename}' (Mode: '{mode}'): {e}")
        print(f"[Error] Unexpected error loading {model_path}: {type(e).__name__} - {e}")
        return None

def get_wordcloud_paths(category_name):
    """Gets the file paths for the word cloud images."""
    path_formatted_name = format_category_for_wordcloud_path(category_name)
    category_folder = os.path.join(WORDCLOUDS_DIR, path_formatted_name)
    low_score_filename = f"low_score_{path_formatted_name}.png"
    high_score_filename = f"high_score_{path_formatted_name}.png"
    low_score_img_path = os.path.join(category_folder, low_score_filename)
    high_score_img_path = os.path.join(category_folder, high_score_filename)

    print(f"[Word Cloud] Checking paths for '{category_name}': Low='{low_score_img_path}', High='{high_score_img_path}'")
    final_low_path = low_score_img_path if os.path.exists(low_score_img_path) else None
    final_high_path = high_score_img_path if os.path.exists(high_score_img_path) else None
    if not final_low_path: print(f"[Warning] Low score image NOT FOUND for {category_name} at {low_score_img_path}")
    if not final_high_path: print(f"[Warning] High score image NOT FOUND for {category_name} at {high_score_img_path}")
    return final_low_path, final_high_path

# --- Define Categories (Ensure consistency) ---
CATEGORIES = sorted([
    'Soups & Chilis', 'Breads & Muffins', 'Breakfast', 'Cakes & Cupcakes',
    'Main Dishes', 'Pies & Tarts', 'Casseroles & Bakes', 'Pasta & Lasagna',
    'Salads & Sides', 'Dessert Specialties', 'Bars & Cookies'
])

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Recipe Score Predictor")

st.title("🍲 Recipe Score Predictor & Analysis")
st.markdown("""
Select a recipe category, input interaction metrics, and choose a prediction model type
(with or without Thumbs Up/Down features) to predict the recipe's 'Best Score'.
""")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("⚙️ Prediction Inputs")

    # --- NEW: Model Mode Selection ---
    model_mode = st.radio(
        "Select Model Type:",
        options=["With Thumbs Up/Down", "Without Thumbs Up/Down"],
        index=0, # Default to the full model
        key="model_mode_selector",
        help="Choose which set of features the predictive model should use."
    )
    st.markdown("---")

    selected_category = st.selectbox(
        "Select Recipe Category:",
        options=CATEGORIES,
        index=0,
        key="category_selector"
    )
    st.markdown("---")

    # --- All Input Widgets Remain ---
    user_reputation = st.number_input("User Reputation", min_value=0, value=1, step=1, key="reputation_input", help="Reputation score of the user.")
    reply_count = st.number_input("Number of Replies", min_value=0, value=0, step=1, key="reply_input")
    thumbs_up = st.number_input("Thumbs Up Count 👍", min_value=0, value=5, step=1, key="thumbs_up_input")
    thumbs_down = st.number_input("Thumbs Down Count 👎", min_value=0, value=0, step=1, key="thumbs_down_input")
    stars = st.slider("Star Rating ⭐", min_value=0, max_value=5, value=4, step=1, key="stars_slider")

    st.markdown("---")
    if model_mode == "With Thumbs Up/Down":
        st.info("Prediction using model trained on: User Reputation, Replies, Thumbs Up, Thumbs Down, Stars.")
    else:
        st.info("Prediction using model trained on: User Reputation, Replies, Stars.")


# --- Main Area Results ---
if selected_category:
    st.header(f"Analysis for: {selected_category}")
    st.subheader(f"Using Model: {model_mode}")

    # --- Prediction Section ---
    st.write("---") # Separator
    st.subheader("📈 Score Prediction")

    # Load the appropriate model based on selection
    model = load_model(selected_category, model_mode)

    if model:
        # Prepare input data based on the selected mode
        input_data_dict = {
            'user_reputation': user_reputation,
            'reply_count': reply_count,
            'thumbs_up': thumbs_up,
            'thumbs_down': thumbs_down,
            'stars': stars
        }

        if model_mode == "With Thumbs Up/Down":
            features_to_use = EXPECTED_FEATURES_FULL
            input_values = [[input_data_dict[feat] for feat in features_to_use]] # Ensure correct order
            input_data = pd.DataFrame(input_values, columns=features_to_use)
            st.write("Input Features (Model: With Thumbs Up/Down):")

        elif model_mode == "Without Thumbs Up/Down":
            features_to_use = EXPECTED_FEATURES_NO_THUMBS
            input_values = [[input_data_dict[feat] for feat in features_to_use]] # Ensure correct order
            input_data = pd.DataFrame(input_values, columns=features_to_use)
            st.write("Input Features (Model: Without Thumbs Up/Down):")
        else:
            # Should not happen if mode selection is correct
             st.error("Internal error: Invalid model mode detected during data preparation.")
             input_data = None

        if input_data is not None:
            st.dataframe(input_data.style.format("{:.0f}"))

            try:
                # Make prediction
                predicted_score = model.predict(input_data)[0]
                st.metric(label=f"Predicted Best Score ({model_mode})", value=f"{predicted_score:.2f}")

                if SCORE_THRESHOLD is not None:
                    score_comparison = "Above or Equal To" if predicted_score >= SCORE_THRESHOLD else "Below"
                    st.info(f"The predicted score is **{score_comparison}** the example threshold ({SCORE_THRESHOLD}).")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.error("Verify model compatibility and input data.")
                print(f"[Error] Prediction step failed for category '{selected_category}', mode '{model_mode}': {e}")
                print(f"Input data causing error: {input_data.to_dict()}")
    else:
        st.warning(f"Prediction cannot proceed. Model for '{selected_category}' (Mode: '{model_mode}') could not be loaded.")
        st.info("Check if the required model file exists in the correct directory and that the environments match.")

    # --- Word Cloud Section (Remains the same) ---
    st.write("---") # Separator
    st.subheader("💬 Word Clouds")
    st.markdown(f"Common words in reviews for **{selected_category}** (Example threshold: {SCORE_THRESHOLD}).")
    low_score_img_path, high_score_img_path = get_wordcloud_paths(selected_category)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Reviews with Score < {SCORE_THRESHOLD}**")
        if low_score_img_path:
            st.image(low_score_img_path, caption=f"Low Score (<{SCORE_THRESHOLD}) Words", use_container_width=True)
        else:
            st.info("Low score word cloud image not found.")
    with col2:
        st.markdown(f"**Reviews with Score >= {SCORE_THRESHOLD}**")
        if high_score_img_path:
            st.image(high_score_img_path, caption=f"High Score (>= {SCORE_THRESHOLD}) Words", use_container_width=True)
        else:
            st.info("High score word cloud image not found.")
else:
    st.warning("Please select a recipe category from the sidebar.")