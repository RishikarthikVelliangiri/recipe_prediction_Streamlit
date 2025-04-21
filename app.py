# streamlit_app.py (or your filename)
import streamlit as st
import pandas as pd
import joblib
import os
import re
import sys

# --- Configuration ---
MODELS_DIR_FULL = 'prediction_models'
MODELS_DIR_NO_THUMBS = 'prediction_models_no_thumbs'
WORDCLOUDS_DIR = 'wordclouds_by_category'
SCORE_THRESHOLD = 200

# --- IMPORTANT: Feature Order Definitions ---
# This order MUST exactly match FEATURE_ORDER in the corresponding generation script
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

@st.cache_resource
def load_model(category_name, mode):
    """Loads the pre-trained model based on category and selected mode."""
    model_path = None # Initialize
    expected_features_list = None
    model_dir = None

    if mode == "With Thumbs Up/Down":
        model_dir = MODELS_DIR_FULL
        expected_features_list = EXPECTED_FEATURES_FULL
    elif mode == "Without Thumbs Up/Down":
        model_dir = MODELS_DIR_NO_THUMBS
        expected_features_list = EXPECTED_FEATURES_NO_THUMBS
    else:
        st.error(f"Internal Error: Invalid model mode selected: {mode}")
        return None, None # Return None for model and expected features

    if not model_dir or not expected_features_list: # Should not happen with valid mode
        return None, None

    expected_features_len = len(expected_features_list)
    model_formatted_name = clean_category_name_for_model_files(category_name)
    model_filename = f"{model_formatted_name}.pkl"
    model_path = os.path.join(model_dir, model_filename)

    print(f"[Model Load] Mode: '{mode}'. Attempting to load model from path: {model_path}")
    print(f"[Model Load] Expecting features: {expected_features_list}") # Print expected features

    try:
        if not os.path.exists(model_path):
             st.error(f"Model file not found for mode '{mode}': '{model_path}'.")
             st.info(f"Ensure you have run the correct model generation script and the file exists in the '{model_dir}' folder. You may need to delete the folder and regenerate.")
             print(f"[Error] Model file does not exist at expected path: {model_path}")
             return None, None

        model = joblib.load(model_path)
        print(f"[Model Load] Successfully loaded model for '{category_name}' (Mode: '{mode}').")

        # --- Rigorous Feature Check ---
        model_expects_n = None
        model_expects_names = None
        if hasattr(model, 'n_features_in_'):
            model_expects_n = model.n_features_in_
        if hasattr(model, 'feature_names_in_'):
             # Convert numpy array to list for easier comparison if needed
             model_expects_names = model.feature_names_in_.tolist()

        print(f"[Model Check] Model reports requiring {model_expects_n} features.")
        if model_expects_names:
             print(f"[Model Check] Model reports feature names: {model_expects_names}")

        if model_expects_n is not None and model_expects_n != expected_features_len:
              st.error(f"Model Feature Count Mismatch (Mode: '{mode}')!")
              st.error(f"- App expects {expected_features_len} features: {expected_features_list}")
              st.error(f"- Loaded model expects {model_expects_n} features.")
              st.warning("This usually means you are loading an OLD or INCORRECT .pkl file. Delete the relevant 'prediction_models*' folder and regenerate the models.")
              print(f"[Error] Feature count mismatch: App={expected_features_len}, Model={model_expects_n}")
              return None, None # Prevent using mismatched model

        # Optional: Check names if available (more robust)
        # if model_expects_names and model_expects_names != expected_features_list:
        #      st.error(f"Model Feature Name/Order Mismatch (Mode: '{mode}')!")
        #      st.error(f"- App expects: {expected_features_list}")
        #      st.error(f"- Model expects: {model_expects_names}")
        #      st.warning("Feature names or order differ! Regenerate models ensuring consistent feature order.")
        #      print(f"[Error] Feature name/order mismatch.")
        #      return None, None

        # Return the model AND the list of features it expects
        return model, expected_features_list

    except (AttributeError, ModuleNotFoundError, ImportError, EOFError, TypeError, ValueError) as load_err:
         st.error(f"Model Load Error for '{category_name}' (Mode: '{mode}'): {type(load_err).__name__} - {load_err}")
         st.error("Check Python/library versions or potential .pkl file corruption.")
         st.warning(f"Ensure 'requirements.txt' matches the environment used for '{model_dir}'. Re-generating .pkl files might fix this.")
         print(f"[Error] Error loading {model_path}: {type(load_err).__name__} - {load_err}")
         return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred loading model '{model_filename}' (Mode: '{mode}'): {e}")
        print(f"[Error] Unexpected error loading {model_path}: {type(e).__name__} - {e}")
        return None, None

def get_wordcloud_paths(category_name):
    """Gets the file paths for the word cloud images."""
    # (Function remains unchanged)
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

# --- Define Categories ---
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

    model_mode = st.radio(
        "Select Model Type:",
        options=["With Thumbs Up/Down", "Without Thumbs Up/Down"],
        index=0,
        key="model_mode_selector",
        help="Choose which features the model should use."
    )
    st.markdown("---")

    selected_category = st.selectbox(
        "Select Recipe Category:",
        options=CATEGORIES,
        index=0,
        key="category_selector"
    )
    st.markdown("---")

    # --- All Inputs Declared (Thumbs Up/Down conditionally displayed) ---
    user_reputation = st.number_input("User Reputation", min_value=0, value=1, step=1, key="reputation_input", help="Reputation score of the user.")
    reply_count = st.number_input("Number of Replies", min_value=0, value=0, step=1, key="reply_input")

    # --- Conditional UI for Thumbs Up/Down ---
    thumbs_up_value = 5 # Default value if not displayed
    thumbs_down_value = 0 # Default value if not displayed
    if model_mode == "With Thumbs Up/Down":
        thumbs_up_value = st.number_input("Thumbs Up Count 👍", min_value=0, value=thumbs_up_value, step=1, key="thumbs_up_input")
        thumbs_down_value = st.number_input("Thumbs Down Count 👎", min_value=0, value=thumbs_down_value, step=1, key="thumbs_down_input")
    # If mode is "Without...", these widgets are simply not drawn.

    stars = st.slider("Star Rating ⭐", min_value=0, max_value=5, value=4, step=1, key="stars_slider")

    # --- Dynamic Info Message ---
    st.markdown("---")
    if model_mode == "With Thumbs Up/Down":
        st.info(f"Using features: {', '.join(EXPECTED_FEATURES_FULL)}")
    else:
        st.info(f"Using features: {', '.join(EXPECTED_FEATURES_NO_THUMBS)}")


# --- Main Area Results ---
if selected_category:
    st.header(f"Analysis for: {selected_category}")
    st.subheader(f"Using Model: {model_mode}")

    # --- Prediction Section ---
    st.write("---")
    st.subheader("📈 Score Prediction")

    # Load model AND the list of features it expects
    model, expected_features = load_model(selected_category, model_mode)

    # Proceed only if model loading was successful AND we know the expected features
    if model and expected_features:
        # Prepare input data dictionary using values from widgets
        input_data_dict = {
            'user_reputation': user_reputation,
            'reply_count': reply_count,
            'thumbs_up': thumbs_up_value, # Use the stored value
            'thumbs_down': thumbs_down_value, # Use the stored value
            'stars': stars
        }

        # Create DataFrame using ONLY the expected features for the loaded model
        try:
            # Select values from dict based on the expected_features list from load_model
            input_values = [[input_data_dict[feat] for feat in expected_features]]
            input_data = pd.DataFrame(input_values, columns=expected_features) # Critical: columns match expected_features

            st.write(f"Input Features Sent to Model ({model_mode}):")
            st.dataframe(input_data.style.format("{:.0f}"))

            # --- Prediction ---
            print(f"[Predict] Calling predict with DataFrame columns: {input_data.columns.tolist()}") # DEBUG Print
            predicted_score = model.predict(input_data)[0]
            st.metric(label=f"Predicted Best Score ({model_mode})", value=f"{predicted_score:.2f}")

            if SCORE_THRESHOLD is not None:
                score_comparison = "Above or Equal To" if predicted_score >= SCORE_THRESHOLD else "Below"
                st.info(f"The predicted score is **{score_comparison}** the example threshold ({SCORE_THRESHOLD}).")

        except KeyError as e:
             st.error(f"Internal Data Preparation Error: Feature '{e}' not found in input dictionary.")
             print(f"[Error] Feature mismatch during DataFrame creation: {e}. Available keys: {input_data_dict.keys()}")
        except Exception as e:
            st.error(f"Error during prediction: {type(e).__name__} - {e}")
            st.error("This could be due to incompatible model versions or unexpected data. Check logs.")
            print(f"[Error] Prediction step failed for category '{selected_category}', mode '{model_mode}': {type(e).__name__} - {e}")
            if 'input_data' in locals(): # Print dataframe if it was created
                 print(f"Input data causing error: {input_data.to_dict()}")

    else:
        # Message if model loading failed
        st.warning(f"Prediction cannot proceed. Model for '{selected_category}' (Mode: '{model_mode}') could not be loaded or verified.")
        st.info("Check the error messages above. Ensure the required model files exist in the correct directory and are compatible.")

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