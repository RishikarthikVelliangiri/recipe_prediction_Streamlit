# %%-----------------------------------------------------------------------------
# Feature Importance Calculation for Multiple Categories in a Grid
# ------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import math # Needed for ceiling division for grid layout

try:
    df_clean = pd.read_csv('cleaned_recipes.csv')
    print("Loaded cleaned_recipes.csv for multi-category analysis.")
except FileNotFoundError:
    print("Error: 'cleaned_recipes.csv' not found.")
    # exit() # Optionally exit or handle the error appropriately

# --- Configuration ---
categories_to_plot = [
    'Breads & Muffins',
    'Breakfast',
    'Cakes & Cupcakes',
    'Casseroles & Bakes',
    'Dessert Specialties',
    'Main Dishes',
    'Pasta & Lasagna',
    'Pies & Tarts',
    'Salads & Sides',
    'Soups & Chilis',
    'Bars & Cookies' # Adding this back as it was analyzed before
]
categories_to_plot.sort() # Optional: Sort alphabetically for consistent grid order

# Features to use for modeling within each category subset
feature_cols_numeric = ['user_reputation', 'reply_count', 'thumbs_up', 'thumbs_down', 'stars']
target_col = 'best_score'

# Grid layout parameters
n_categories = len(categories_to_plot)
n_cols = 3 # Adjust number of columns as desired
n_rows = math.ceil(n_categories / n_cols) # Calculate required rows

# --- Create Figure and Axes for the Grid ---
fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=False) # sharex=False might be better if importance scales vary
axs = axs.flatten() # Flatten the axes array for easy iteration

print(f"Preparing {n_rows}x{n_cols} grid for {n_categories} categories...")

# --- Loop Through Categories, Train Models, and Plot ---
plot_index = 0
for category_name in categories_to_plot:
    print(f"\nProcessing Category: {category_name}...")

    # Filter the DataFrame
    df_filtered = df_clean[df_clean['category'] == category_name].copy()

    ax = axs[plot_index] # Get the current subplot axis

    # Check if enough data exists
    min_samples_for_plot = 10 # Set a minimum threshold
    if len(df_filtered) < min_samples_for_plot:
        print(f"Skipping '{category_name}': Only {len(df_filtered)} samples (less than minimum {min_samples_for_plot}).")
        ax.set_title(f"{category_name}\n(Not enough data)", fontsize=10)
        ax.axis('off') # Turn off axis for skipped plots
        plot_index += 1
        continue # Move to the next category

    print(f"Found {len(df_filtered)} samples.")

    # Select features and target
    # Check if columns exist (redundant if checked before, but safe)
    missing_cols = [col for col in feature_cols_numeric + [target_col] if col not in df_filtered.columns]
    if missing_cols:
         print(f"Error: Columns missing for '{category_name}': {missing_cols}")
         ax.set_title(f"{category_name}\n(Data error)", fontsize=10)
         ax.axis('off')
         plot_index += 1
         continue

    X_filtered = df_filtered[feature_cols_numeric]
    y_filtered = df_filtered[target_col]

    # Train Model
    model_filtered = RandomForestRegressor(n_estimators=100,
                                           random_state=42,
                                           n_jobs=-1,
                                           min_samples_leaf=3) # Keep some regularization
    try:
        model_filtered.fit(X_filtered, y_filtered)

        # Get Feature Importances
        importances_filtered = model_filtered.feature_importances_
        feature_importance_filtered_df = pd.DataFrame({
            'Feature': feature_cols_numeric,
            'Importance': importances_filtered
        }).sort_values(by='Importance', ascending=False)

        # Plot on the specific subplot axis
        sns.barplot(x='Importance', y='Feature', data=feature_importance_filtered_df, ax=ax, palette='viridis')
        ax.set_title(f"{category_name} ({len(df_filtered)} rows)", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8) # Adjust label size
        ax.set_xlabel("Importance", fontsize=9)
        ax.set_ylabel("") # Remove y-label to reduce clutter

        # Optional: Add grid lines to each subplot
        ax.grid(axis='x', linestyle='--', alpha=0.6)

    except Exception as e:
         print(f"Error during model training/plotting for '{category_name}': {e}")
         ax.set_title(f"{category_name}\n(Training Error)", fontsize=10)
         ax.axis('off') # Turn off axis if error occurs

    plot_index += 1 # Move to the next subplot position

# --- Final Touches ---
# Hide any unused subplots at the end of the grid
for i in range(plot_index, n_rows * n_cols):
    axs[i].axis('off')

plt.suptitle('Feature Importance by Category for Best Score Prediction', fontsize=16, y=1.02) # Add overall title
plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent title overlap and make space
plt.show()

print("\n--- Analysis Complete ---")