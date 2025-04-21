# generate_importance_plots.py (Corrected for "No Thumbs Up")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
# joblib is not needed here unless loading external data/models
import os
import math
import re # Keep re for filename cleaning function consistency

print("--- Starting Feature Importance Plot Generation ---")

# --- Configuration ---
CLEANED_DATA_FILE = 'cleaned_recipes.csv'
OUTPUT_PLOT_DIR = 'feature_importance_plots' # Folder to save the grid plots
TARGET_COL = 'best_score'
CATEGORY_COL = 'category'

# Define the THREE sets of features to analyze
FEATURE_SETS = [
    {
        "name": "All Features (incl. Thumbs)",
        "features": ['user_reputation', 'reply_count', 'thumbs_up', 'thumbs_down', 'stars'],
        "filename": "importance_grid_all_features.png"
    },
    # --- MODIFIED THIS ENTRY ---
    {
        "name": "No Thumbs Up", # Changed Name
        "features": ['user_reputation', 'reply_count', 'thumbs_down', 'stars'], # Removed thumbs_up, kept thumbs_down
        "filename": "importance_grid_no_thumbs_up.png" # Changed Filename
    },
    # --- END MODIFICATION ---
    {
        "name": "No Thumbs Up or Down",
        "features": ['user_reputation', 'reply_count', 'stars'],
        "filename": "importance_grid_no_thumbs_both.png"
    }
]

# Plotting Configuration
N_COLS_GRID = 3 # Number of columns in the plot grid
MIN_SAMPLES_FOR_PLOT = 10 # Minimum samples required per category to generate its plot

# --- Load Data ---
try:
    df_full = pd.read_csv(CLEANED_DATA_FILE)
    print(f"Loaded data from '{CLEANED_DATA_FILE}' ({len(df_full)} rows)")
except FileNotFoundError:
    print(f"Error: Cleaned data file not found at '{CLEANED_DATA_FILE}'")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Prepare Data ---
all_needed_cols = list(set([TARGET_COL, CATEGORY_COL] + [feat for aset in FEATURE_SETS for feat in aset['features']]))
print(f"Using columns: {all_needed_cols}")

try:
    df = df_full[all_needed_cols].copy()
except KeyError as e:
    print(f"Error: Column {e} not found in the dataset. Required columns: {all_needed_cols}")
    exit()

initial_rows = len(df)
df = df.dropna()
print(f"Shape after dropping NaNs: {df.shape} (Removed {initial_rows - len(df)} rows)")

categories = sorted(df[CATEGORY_COL].unique())
print(f"Found {len(categories)} categories to process.")

# --- Create Output Directory ---
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
print(f"Plots will be saved to: '{OUTPUT_PLOT_DIR}'")

# Define the filename cleaning function (consistent with other scripts)
def clean_category_name_for_files(category_name):
    name = category_name.replace(' & ', '_and_')
    name = re.sub(r'[^\w\-\.]', '_', name)
    name = re.sub(r'_+', '_', name)
    return name

# --- Main Loop: Iterate through each Feature Set ---
for feature_set_info in FEATURE_SETS:
    set_name = feature_set_info["name"]
    current_features = feature_set_info["features"]
    output_filename = feature_set_info["filename"]
    output_filepath = os.path.join(OUTPUT_PLOT_DIR, output_filename)

    print(f"\n--- Generating Plot Grid for: {set_name} ---")
    print(f"Using features: {current_features}")

    # --- Setup Grid Plot ---
    n_categories = len(categories)
    n_rows = math.ceil(n_categories / N_COLS_GRID)
    fig, axs = plt.subplots(n_rows, N_COLS_GRID, figsize=(5 * N_COLS_GRID, 4.5 * n_rows), sharex=False)
    axs = axs.flatten() # Flatten for easy iteration

    plot_index = 0
    # --- Inner Loop: Iterate through each Category ---
    for category_name in categories:
        print(f"  Processing Category: {category_name}...")
        ax = axs[plot_index] # Get the current subplot axis

        # Filter data for this category
        category_df = df[df[CATEGORY_COL] == category_name].copy()

        # Check if enough data exists
        if len(category_df) < MIN_SAMPLES_FOR_PLOT:
            print(f"  Skipping '{category_name}': Only {len(category_df)} samples.")
            ax.set_title(f"{category_name}\n(Not enough data)", fontsize=9)
            ax.axis('off')
            plot_index += 1
            continue

        # Prepare X and y using CURRENT feature set
        try:
            X = category_df[current_features]
            y = category_df[TARGET_COL]
        except KeyError as e:
            print(f"  Error selecting features for '{category_name}': Missing {e}. Skipping.")
            ax.set_title(f"{category_name}\n(Data error)", fontsize=9)
            ax.axis('off')
            plot_index += 1
            continue

        # Train a simple model for importance calculation
        try:
            model = RandomForestRegressor(n_estimators=100,
                                          random_state=42,
                                          n_jobs=-1,
                                          min_samples_leaf=3) # Regularization
            model.fit(X, y)

            # Get Feature Importances
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': current_features,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            # Plot on the specific subplot axis
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, palette='viridis')
            ax.set_title(f"{category_name} ({len(category_df)} rows)", fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.set_xlabel("Importance", fontsize=9)
            ax.set_ylabel("") # Reduce clutter
            ax.grid(axis='x', linestyle='--', alpha=0.6)

        except Exception as e:
             print(f"  Error during model training/plotting for '{category_name}': {e}")
             ax.set_title(f"{category_name}\n(Plotting Error)", fontsize=10)
             ax.axis('off')

        plot_index += 1 # Move to the next subplot position

    # --- Finalize Grid Plot ---
    for i in range(plot_index, n_rows * N_COLS_GRID): # Hide unused
        try: axs[i].axis('off')
        except IndexError: pass
    fig.suptitle(f'Feature Importance by Category\n({set_name})', fontsize=16, y=1.03)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    # --- Save the Figure ---
    try:
        plt.savefig(output_filepath, dpi=150, bbox_inches='tight')
        print(f"--> Saved plot grid to: {output_filepath}")
    except Exception as e:
        print(f"Error saving plot '{output_filepath}': {e}")
    plt.close(fig)

print("\n--- Feature Importance Plot Generation Complete ---")