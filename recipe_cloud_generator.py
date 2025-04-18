import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os
import re
import string

# --- Configuration ---
CSV_FILE = 'cleaned_recipes.csv'
OUTPUT_DIR = 'wordclouds_by_category'
SCORE_THRESHOLD = 200
TEXT_COLUMN = 'text'
CATEGORY_COLUMN = 'category'
SCORE_COLUMN = 'best_score'

# --- Create Output Directory ---
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory '{OUTPUT_DIR}' ensured.")
except OSError as e:
    print(f"Error creating directory {OUTPUT_DIR}: {e}")
    exit() # Exit if we can't create the output directory

# --- Load Data ---
try:
    df = pd.read_csv(CSV_FILE)
    print(f"Successfully loaded data from '{CSV_FILE}'.")
except FileNotFoundError:
    print(f"Error: The file '{CSV_FILE}' was not found.")
    exit()
except Exception as e:
    print(f"An error occurred while reading the CSV: {e}")
    exit()

# --- Data Cleaning and Preparation ---
print("Preparing data...")

# Ensure necessary columns exist
required_columns = [TEXT_COLUMN, CATEGORY_COLUMN, SCORE_COLUMN]
if not all(col in df.columns for col in required_columns):
    print(f"Error: CSV must contain the columns: {', '.join(required_columns)}")
    missing = [col for col in required_columns if col not in df.columns]
    print(f"Missing columns: {', '.join(missing)}")
    exit()

# Convert score column to numeric, handling potential errors
df[SCORE_COLUMN] = pd.to_numeric(df[SCORE_COLUMN], errors='coerce')

# Handle missing text or category values
df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna('').astype(str)
df[CATEGORY_COLUMN] = df[CATEGORY_COLUMN].fillna('Unknown').astype(str)

# Drop rows where score is not valid or category is missing (if desired - here we handle unknown category)
initial_rows = len(df)
df.dropna(subset=[SCORE_COLUMN], inplace=True)
if len(df) < initial_rows:
    print(f"Dropped {initial_rows - len(df)} rows due to invalid scores.")

if df.empty:
    print("Error: No valid data remaining after cleaning.")
    exit()

# --- Define Stopwords ---
# Start with default stopwords and add custom ones
custom_stopwords = set(STOPWORDS)
custom_stopwords.update([
    'recipe', 'make', 'made', 'use', 'used', 'time', 'cook', 'cooked',
    'add', 'added', 'cup', 'cups', 'teaspoon', 'tablespoon', 'ingredient',
    'ingredients', 'also', 'really', 'just', 'like', 'great', 'good',
    'easy', 'simple', 'will', 'try', 'instead', 'little', 'bit', 'well',
    'one', 'first', 'next', 'thank', 'thanks', 'flavor', 'texture',
    'recommend', 'definitely', 'family', 'loved', 'love', 'husband',
    'delicious', 'amazing', 'perfect', 'served', 'didn', 'don', 'didn\'t', 'don\'t',
    'came', 'turned', 'found', 'sure', 'make', 'making', 'going', 'tried'
    # Add more words specific to your recipes/comments if needed
])

# --- Helper Function for Text Cleaning and Word Cloud Generation ---
def generate_wordcloud_for_group(data_subset, title_suffix, filename_prefix, category_name, stopwords_set):
    """
    Cleans text, generates, saves, and plots a word cloud for a DataFrame subset.
    """
    print(f"\nProcessing: {category_name} - {title_suffix}")

    if data_subset.empty:
        print(f"  Skipping: No data for '{category_name}' with {title_suffix}.")
        return

    # Combine all text for the group
    full_text = ' '.join(data_subset[TEXT_COLUMN].astype(str))

    # Clean the text
    text = full_text.lower() # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace

    if not text:
        print(f"  Skipping: No text content found for '{category_name}' with {title_suffix} after cleaning.")
        return

    try:
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=stopwords_set,
            min_font_size=10,
            max_words=150, # Limit number of words
            collocations=False # Avoid bi-grams for simplicity
        ).generate(text)

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"{category_name} - {title_suffix}", fontsize=14)

        # Save the image
        # Clean category name for filename (remove special chars)
        safe_category_name = re.sub(r'[^\w\-_\. ]', '_', category_name)
        category_dir = os.path.join(OUTPUT_DIR, safe_category_name)
        os.makedirs(category_dir, exist_ok=True) # Ensure category sub-directory exists

        filename = f"{filename_prefix}_{safe_category_name}.png"
        save_path = os.path.join(category_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close() # Close the plot to free memory
        print(f"  Saved word cloud to: {save_path}")

    except ValueError as ve:
         print(f"  Skipping due to ValueError (likely empty text after stopword removal) for '{category_name}' - {title_suffix}: {ve}")
    except Exception as e:
        print(f"  An error occurred generating word cloud for '{category_name}' - {title_suffix}: {e}")


# --- Main Loop: Iterate Through Categories ---
unique_categories = df[CATEGORY_COLUMN].unique()
print(f"\nFound {len(unique_categories)} unique categories. Generating word clouds...")

for category in unique_categories:
    if not category or pd.isna(category):
        print("Skipping empty or NaN category.")
        continue

    print(f"\n----- Processing Category: {category} -----")

    # Filter data for the current category
    category_df = df[df[CATEGORY_COLUMN] == category].copy() # Use .copy() to avoid SettingWithCopyWarning

    if category_df.empty:
        print(f"  No data found for category: {category}")
        continue

    # Split data based on score threshold
    low_score_df = category_df[category_df[SCORE_COLUMN] < SCORE_THRESHOLD]
    high_score_df = category_df[category_df[SCORE_COLUMN] >= SCORE_THRESHOLD]

    # --- Generate Word Cloud for Low Score Recipes ---
    generate_wordcloud_for_group(
        data_subset=low_score_df,
        title_suffix=f'Score < {SCORE_THRESHOLD}',
        filename_prefix='low_score',
        category_name=category,
        stopwords_set=custom_stopwords
    )

    # --- Generate Word Cloud for High Score Recipes ---
    generate_wordcloud_for_group(
        data_subset=high_score_df,
        title_suffix=f'Score >= {SCORE_THRESHOLD}',
        filename_prefix='high_score',
        category_name=category,
        stopwords_set=custom_stopwords
    )

print("\n----- Word cloud generation process completed. -----")