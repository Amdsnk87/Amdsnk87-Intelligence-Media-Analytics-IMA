import nltk
import os

# Create NLTK data directory if it doesn't exist
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
    print(f"Created NLTK data directory at {nltk_data_dir}")

# Download required NLTK resources
print("Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')  # Add punkt_tab resource
nltk.download('averaged_perceptron_tagger')
print("NLTK resources downloaded successfully!")