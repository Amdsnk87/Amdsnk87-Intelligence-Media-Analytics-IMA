#!/bin/bash

# Install the en_core_web_sm model for SpaCy
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger')"

# Make the script executable
chmod +x setup.sh