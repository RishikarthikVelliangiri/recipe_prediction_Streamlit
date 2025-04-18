import streamlit as st
import pandas as pd
import joblib
import os
import re
import sys # Keep for potential future debugging if needed

# --- Configuration ---
MODELS_DIR = 'prediction_models'
WORDCLOUDS_DIR = 'wordclouds_by_category'
SCORE_THRESHOLD = 200
# Ensure this order matches the features your models were trained on
EXPECTED_FEATURES = ['reply_count', 'thumbs_up', 'thumbs_down', 'stars']

# --- Helper Functions ---

def clean_category_name_for_model_files(category_name):
    """Cleans category name to match the MODEL filename convention."""
    name = category_name.replace(' & ', '_and_')
    # Replace non-alphanumeric (allow _, -, .)
    name = re.sub(r'[^\w\-\.]', '_', name)
    # Consolidate underscores
    name = re.sub(r'_+', '_', name)
    return name

def format_category_for_wordcloud_path(category_name):
    """Formats category name to match the WORD CLOUD folder/file convention."""
    # Handle ' & ' specifically
    name = category_name.replace(' & ', ' _ ')
    # Assuming single names like 'Breakfast' or 'Dessert Specialties' don't need formatting
    return name

# Use Streamlit's caching for models to avoid reloading on every interaction
@st.cache_resource
def load_model(category_name):
    """Loads the pre-trained model for the selected category."""
    model_formatted_name = clean_category_name_for_model_files(category_name)
    model_filename = f"{model_formatted_name}.pkl"
    model_path = os.path.join(MODELS_DIR, model_filename)

    # Log attempt to backend/terminal (useful for Streamlit Cloud logs)
    print(f"[Model Load] Attempting to load model from path: {model_path}")

    try:
        # Check existence first
        if not os.path.exists(model_path):
             st.error(f"Model file not found: '{model_path}'. Check repository.")
             print(f"[Error] Model file does not exist at expected path: {model_path}")
             return None

        # Load the model
        model = joblib.load(model_path)
        print(f"[Model Load] Successfully loaded model for '{category_name}'.")
        return model

    # Handle potential version/corruption errors during loading
    except (AttributeError, ModuleNotFoundError, ImportError) as version_err:
         st.error(f"Model Load Error for '{category_name}': {version_err}")
         st.error("This likely means a mismatch between the environment used to save the model (.pkl file) and the environment running this app (Python or library versions like NumPy/Scikit-learn).")
         st.warning("Ensure 'requirements.txt' matches the model saving environment. Re-saving the model might be necessary.")
         print(f"[Error] Version mismatch or import error loading {model_path}: {version_err}")
         return None
    # Handle other potential errors during loading
    except Exception as e:
        st.error(f"An unexpected error occurred loading model '{model_filename}': {e}")
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

    # Log checks to backend/terminal
    print(f"[Word Cloud] Checking paths for '{category_name}': Low='{low_score_img_path}', High='{high_score_img_path}'")

    # Return paths if they exist, otherwise None
    final_low_path = low_score_img_path if os.path.exists(low_score_img_path) else None
    final_high_path = high_score_img_path if os.path.exists(high_score_img_path) else None

    if not final_low_path: print(f"[Warning] Low score image NOT FOUND for {category_name}")
    if not final_high_path: print(f"[Warning] High score image NOT FOUND for {category_name}")

    return final_low_path, final_high_path

# --- Define Categories (Ensure these match user expectations and your data) ---
CATEGORIES = sorted([ # Sort alphabetically for consistency
    'Soups & Chilis', 'Breads & Muffins', 'Breakfast', 'Cakes & Cupcakes',
    'Main Dishes', 'Pies & Tarts', 'Casseroles & Bakes', 'Pasta & Lasagna',
    'Salads & Sides', 'Dessert Specialties', 'Bars & Cookies'
])

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Recipe Score Predictor")

st.title("🍲 Recipe Score Predictor & Analysis")
st.markdown("""
Select a recipe category and input review interaction metrics to predict the recipe's likely
'Best Score' (a popularity/ranking indicator). You can also view word clouds showing common
terms found in low-scoring vs. high-scoring reviews for the selected category.
""")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("⚙️ Prediction Inputs")
    selected_category = st.selectbox(
        "Select Recipe Category:",
        options=CATEGORIES,
        index=0, # Default to first category
        key="category_selector"
    )
    st.markdown("---") # Separator
    reply_count = st.number_input("Number of Replies", min_value=0, value=0, step=1, key="reply_input")
    thumbs_up = st.number_input("Thumbs Up Count 👍", min_value=0, value=5, step=1, key="thumbs_up_input")
    thumbs_down = st.number_input("Thumbs Down Count 👎", min_value=0, value=0, step=1, key="thumbs_down_input")
    stars = st.slider("Star Rating ⭐", min_value=0, max_value=5, value=4, step=1, key="stars_slider")
    st.markdown("---")
    st.info("App based on recipe review data analysis.")

# --- Main Area Results ---
if selected_category:
    st.header(f"Analysis for: {selected_category}")

    # --- Prediction Section ---
    st.subheader("📈 Score Prediction")
    # Attempt to load the model; errors handled within the function
    model = load_model(selected_category)

    if model:
        # Prepare input data only if model loaded
        input_data = pd.DataFrame([[reply_count, thumbs_up, thumbs_down, stars]],
                                   columns=EXPECTED_FEATURES)
        st.write("Input Features:")
        st.dataframe(input_data)
        try:
            # Make prediction
            predicted_score = model.predict(input_data)[0]
            st.metric(label="Predicted Best Score", value=f"{predicted_score:.2f}")
            # Compare to threshold
            score_comparison = "Above or Equal To" if predicted_score >= SCORE_THRESHOLD else "Below"
            st.info(f"The predicted score is **{score_comparison}** the threshold ({SCORE_THRESHOLD}).")
        except Exception as e:
            # Catch errors during the predict step
            st.error(f"Error during prediction: {e}")
            print(f"[Error] Prediction step failed: {e}")
    else:
        # Message if model loading failed (error details already shown by load_model)
        st.error("Prediction cannot proceed because the model for this category could not be loaded.")

    # --- Word Cloud Section ---
    st.subheader("💬 Word Clouds")
    st.markdown(f"Common words in reviews for **{selected_category}**.")
    # Get image paths
    low_score_img_path, high_score_img_path = get_wordcloud_paths(selected_category)
    # Display side-by-side
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
    # Fallback if no category is selected (shouldn't happen with default)
    st.warning("Please select a recipe category from the sidebar.")