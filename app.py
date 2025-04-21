# streamlit_app.py (Corrected for "No Thumbs Up" plot)
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
FEATURE_PLOTS_DIR = 'feature_importance_plots'
SCORE_THRESHOLD = 200

# Expected features for BOTH model sets
EXPECTED_FEATURES_FULL = ['user_reputation', 'reply_count', 'thumbs_up', 'thumbs_down', 'stars']
EXPECTED_FEATURES_NO_THUMBS = ['user_reputation', 'reply_count', 'stars']

# --- MODIFIED: Expected filenames for the importance plots ---
PLOT_FILENAME_ALL = "importance_grid_all_features.png"
PLOT_FILENAME_NO_UP = "importance_grid_no_thumbs_up.png" # Changed constant name and value
PLOT_FILENAME_NO_BOTH = "importance_grid_no_thumbs_both.png"
# --- END MODIFICATION ---


# --- Helper Functions ---
# (Keep the existing helper functions: clean_category_name_for_model_files,
#  format_category_for_wordcloud_path, load_model, get_wordcloud_paths)
def clean_category_name_for_model_files(category_name):
    name = category_name.replace(' & ', '_and_')
    name = re.sub(r'[^\w\-\.]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name

def format_category_for_wordcloud_path(category_name):
    name = category_name.replace(' & ', ' _ ')
    return name

@st.cache_resource
def load_model(category_name, mode):
    model_path = None
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
        return None, None
    if not model_dir or not expected_features_list: return None, None

    expected_features_len = len(expected_features_list)
    model_formatted_name = clean_category_name_for_model_files(category_name)
    model_filename = f"{model_formatted_name}.pkl"
    model_path = os.path.join(model_dir, model_filename)
    print(f"[Model Load] Mode: '{mode}'. Attempting to load model from path: {model_path}")
    print(f"[Model Load] Expecting features: {expected_features_list}")
    try:
        if not os.path.exists(model_path):
             st.error(f"Model file not found for mode '{mode}': '{model_path}'.")
             st.info(f"Ensure the file exists in the '{model_dir}' folder.")
             print(f"[Error] Model file does not exist at expected path: {model_path}")
             return None, None
        model = joblib.load(model_path)
        print(f"[Model Load] Successfully loaded model for '{category_name}' (Mode: '{mode}').")
        model_expects_n = None
        if hasattr(model, 'n_features_in_'): model_expects_n = model.n_features_in_
        print(f"[Model Check] Model reports requiring {model_expects_n} features.")
        if model_expects_n is not None and model_expects_n != expected_features_len:
              st.error(f"Model Feature Count Mismatch (Mode: '{mode}')!")
              st.error(f"- App expects {expected_features_len} features: {expected_features_list}")
              st.error(f"- Loaded model expects {model_expects_n} features.")
              st.warning("Delete the relevant 'prediction_models*' folder and regenerate the models.")
              print(f"[Error] Feature count mismatch: App={expected_features_len}, Model={model_expects_n}")
              return None, None
        return model, expected_features_list
    except Exception as e:
        st.error(f"An unexpected error occurred loading model '{model_filename}' (Mode: '{mode}'): {type(e).__name__} - {e}")
        print(f"[Error] Unexpected error loading {model_path}: {type(e).__name__} - {e}")
        return None, None

def get_wordcloud_paths(category_name):
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
# --- End Helper Functions ---

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
to predict the recipe's 'Best Score'. Explore word clouds and feature importance plots below.
""")

# --- Sidebar Inputs ---
with st.sidebar:
    # (Sidebar code remains the same as previous version)
    st.header("⚙️ Prediction Inputs")
    model_mode = st.radio(
        "Select Model Type:",
        options=["With Thumbs Up/Down", "Without Thumbs Up/Down"], index=0,
        key="model_mode_selector", help="Choose which features the model should use."
    )
    st.markdown("---")
    selected_category = st.selectbox(
        "Select Recipe Category:", options=CATEGORIES, index=0, key="category_selector"
    )
    st.markdown("---")
    user_reputation = st.number_input("User Reputation", min_value=0, value=1, step=1, key="reputation_input")
    reply_count = st.number_input("Number of Replies", min_value=0, value=0, step=1, key="reply_input")
    thumbs_up_value = 5
    thumbs_down_value = 0
    if model_mode == "With Thumbs Up/Down":
        thumbs_up_value = st.number_input("Thumbs Up Count 👍", min_value=0, value=thumbs_up_value, step=1, key="thumbs_up_input")
        thumbs_down_value = st.number_input("Thumbs Down Count 👎", min_value=0, value=thumbs_down_value, step=1, key="thumbs_down_input")
    stars = st.slider("Star Rating ⭐", min_value=0, max_value=5, value=4, step=1, key="stars_slider")
    st.markdown("---")
    if model_mode == "With Thumbs Up/Down":
        st.info(f"Using features: {', '.join(EXPECTED_FEATURES_FULL)}")
    else:
        st.info(f"Using features: {', '.join(EXPECTED_FEATURES_NO_THUMBS)}")
# --- End Sidebar ---

# --- Main Area ---
if selected_category:
    st.header(f"Analysis for: {selected_category}")

    # --- Prediction Section ---
    # (Prediction section remains the same as previous version)
    st.subheader(f"📈 Score Prediction (Model: {model_mode})")
    model, expected_features = load_model(selected_category, model_mode)
    if model and expected_features:
        input_data_dict = {'user_reputation': user_reputation,'reply_count': reply_count,'thumbs_up': thumbs_up_value,'thumbs_down': thumbs_down_value,'stars': stars}
        try:
            input_values = [[input_data_dict[feat] for feat in expected_features]]
            input_data = pd.DataFrame(input_values, columns=expected_features)
            st.write(f"Input Features Sent to Model ({model_mode}):")
            st.dataframe(input_data.style.format("{:.0f}"))
            print(f"[Predict] Calling predict with DataFrame columns: {input_data.columns.tolist()}")
            predicted_score = model.predict(input_data)[0]
            st.metric(label=f"Predicted Best Score ({model_mode})", value=f"{predicted_score:.2f}")
            if SCORE_THRESHOLD is not None:
                score_comparison = "Above or Equal To" if predicted_score >= SCORE_THRESHOLD else "Below"
                st.info(f"The predicted score is **{score_comparison}** the example threshold ({SCORE_THRESHOLD}).")
        except Exception as e:
            st.error(f"Error during prediction: {type(e).__name__} - {e}")
            print(f"[Error] Prediction step failed: {type(e).__name__} - {e}")
    else:
        st.warning(f"Prediction cannot proceed. Model for '{selected_category}' (Mode: '{model_mode}') could not be loaded.")
    st.write("---")
    # --- End Prediction Section ---

    # --- Word Cloud Section ---
    # (Word cloud section remains the same as previous version)
    st.subheader("💬 Word Clouds")
    st.markdown(f"Common words in reviews for **{selected_category}**.")
    low_score_img_path, high_score_img_path = get_wordcloud_paths(selected_category)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Reviews with Score < {SCORE_THRESHOLD}**")
        if low_score_img_path: st.image(low_score_img_path, use_container_width=True)
        else: st.info("Low score word cloud image not found.")
    with col2:
        st.markdown(f"**Reviews with Score >= {SCORE_THRESHOLD}**")
        if high_score_img_path: st.image(high_score_img_path, use_container_width=True)
        else: st.info("High score word cloud image not found.")
    st.write("---")
    # --- End Word Cloud Section ---

    # --- Feature Importance Section ---
    st.subheader("📊 Feature Importance Analysis")
    st.markdown("""
    These plots show the relative importance of different features in predicting the 'Best Score'
    across all categories, based on models trained with different feature sets.
    Higher bars indicate features the model relied on more heavily.
    """)

    # Define paths to the generated plot images
    plot_path_all = os.path.join(FEATURE_PLOTS_DIR, PLOT_FILENAME_ALL)
    # --- MODIFIED: Path for the "No Thumbs Up" plot ---
    plot_path_no_up = os.path.join(FEATURE_PLOTS_DIR, PLOT_FILENAME_NO_UP) # Changed variable name and uses updated constant
    # --- END MODIFICATION ---
    plot_path_no_both = os.path.join(FEATURE_PLOTS_DIR, PLOT_FILENAME_NO_BOTH)

    # Display Plot 1: All Features
    st.markdown("**1. Importance: All Features (incl. Thumbs Up/Down)**")
    if os.path.exists(plot_path_all):
        st.image(plot_path_all, caption="Feature importance grid using all main features.")
    else:
        st.warning(f"Plot file not found: '{plot_path_all}'. Please run the 'generate_importance_plots.py' script.")
    st.markdown("---")

    # --- MODIFIED: Display Plot 2: No Thumbs Up ---
    st.markdown("**2. Importance: No Thumbs Up Feature**") # Changed Header
    if os.path.exists(plot_path_no_up): # Use updated path variable
        st.image(plot_path_no_up, caption="Feature importance grid excluding 'thumbs_up'.") # Changed Caption
    else:
        st.warning(f"Plot file not found: '{plot_path_no_up}'. Please run the 'generate_importance_plots.py' script.") # Use updated path variable in warning
    st.markdown("---")
    # --- END MODIFICATION ---

    # Display Plot 3: No Thumbs Up or Down
    st.markdown("**3. Importance: No Thumbs Up or Down Features**")
    if os.path.exists(plot_path_no_both):
        st.image(plot_path_no_both, caption="Feature importance grid excluding both 'thumbs_up' and 'thumbs_down'.")
    else:
        st.warning(f"Plot file not found: '{plot_path_no_both}'. Please run the 'generate_importance_plots.py' script.")
    # --- End Feature Importance Section ---

else:
    st.warning("Please select a recipe category from the sidebar.")
# --- End Main Area ---