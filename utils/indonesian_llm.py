import pandas as pd
import numpy as np
import os
import re
import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from sklearn.model_selection import train_test_split
from datasets import Dataset
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Try to extend stopwords with Indonesian stopwords
try:
    STOPWORDS = set(stopwords.words('indonesian'))
except:
    # Fallback to a basic set of Indonesian stopwords if NLTK doesn't have them
    STOPWORDS = {
        'yang', 'dan', 'di', 'ke', 'pada', 'untuk', 'dari', 'dengan', 'adalah', 'ini',
        'itu', 'atau', 'juga', 'ada', 'akan', 'bisa', 'dalam', 'oleh', 'secara', 'hal',
        'dapat', 'tersebut', 'bahwa', 'para', 'saat', 'harus', 'saya', 'kamu', 'kami',
        'mereka', 'dia', 'kita', 'sedang', 'sudah', 'telah', 'anda', 'sebagai', 'jika',
        'tidak', 'ya', 'karena', 'tahun', 'sejak', 'tentang', 'seperti', 'lebih', 'belum',
        'hari', 'sehingga', 'hingga', 'sebuah', 'setelah', 'tetapi', 'tapi', 'pun', 'besar'
    }

def preprocess_indonesian_text(text):
    """
    Preprocess Indonesian text for model training.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def create_training_dataset(news_df, text_column='content', label_column=None):
    """
    Create a dataset for fine-tuning a language model on Indonesian content.
    
    Args:
        news_df: DataFrame containing news articles
        text_column: Column containing text content
        label_column: Optional column containing labels for classification tasks
        
    Returns:
        Dataset ready for training
    """
    if news_df.empty or text_column not in news_df.columns:
        raise ValueError(f"DataFrame is empty or missing required column: {text_column}")
        
    # Preprocess text
    news_df[f'processed_{text_column}'] = news_df[text_column].apply(preprocess_indonesian_text)
    
    # Create dataset
    if label_column and label_column in news_df.columns:
        # Classification dataset
        train_data = {
            'text': news_df[f'processed_{text_column}'].tolist(),
            'labels': news_df[label_column].tolist()
        }
    else:
        # Language modeling dataset
        train_data = {
            'text': news_df[f'processed_{text_column}'].tolist()
        }
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict(train_data)
    
    return dataset

def finetune_sentiment_classifier(
    news_df,
    text_column='content',
    label_column='sentiment_category',
    model_name='indolem/indobert-base-uncased',
    output_dir='./models/sentiment',
    epochs=3,
    batch_size=8
):
    """
    Fine-tune a sentiment classifier for Indonesian content.
    
    Args:
        news_df: DataFrame with news articles
        text_column: Column containing text
        label_column: Column containing sentiment labels
        model_name: Base model to fine-tune
        output_dir: Output directory for saving model
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Dictionary with training info and model path
    """
    if news_df.empty or text_column not in news_df.columns or label_column not in news_df.columns:
        raise ValueError("DataFrame is empty or missing required columns")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    full_dataset = create_training_dataset(news_df, text_column, label_column)
    
    # Split dataset
    train_test = full_dataset.train_test_split(test_size=0.2)
    
    # Get unique labels and map to IDs
    unique_labels = sorted(set(news_df[label_column].unique()))
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    # Save label mappings
    with open(os.path.join(output_dir, 'label_mapping.json'), 'w') as f:
        json.dump({'label2id': label2id, 'id2label': id2label}, f)
    
    # Convert string labels to IDs
    def map_labels(examples):
        examples['labels'] = [label2id[label] for label in examples['labels']]
        return examples
    
    train_test = train_test.map(map_labels)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id
    )
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = train_test.map(tokenize_function, batched=True)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )
    
    # Train model
    train_result = trainer.train()
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Evaluation
    eval_result = trainer.evaluate()
    
    # Save training info
    training_info = {
        'model_name': model_name,
        'base_model': model_name,
        'fine_tuned_task': 'sentiment_classification',
        'dataset_size': len(full_dataset),
        'train_size': len(tokenized_datasets["train"]),
        'test_size': len(tokenized_datasets["test"]),
        'num_labels': len(unique_labels),
        'labels': unique_labels,
        'training_loss': train_result.training_loss,
        'evaluation_loss': eval_result['eval_loss'],
        'epochs': epochs,
        'date_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_path': output_dir
    }
    
    # Save training info
    with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
        json.dump(training_info, f)
    
    return training_info

def finetune_topic_classifier(
    news_df,
    text_column='content',
    label_column='category',
    model_name='indolem/indobert-base-uncased',
    output_dir='./models/topic',
    epochs=3,
    batch_size=8
):
    """
    Fine-tune a topic classifier for Indonesian content.
    
    Args:
        news_df: DataFrame with news articles
        text_column: Column containing text
        label_column: Column containing topic labels
        model_name: Base model to fine-tune
        output_dir: Output directory for saving model
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Dictionary with training info and model path
    """
    # Implementation is similar to sentiment classifier
    # Reuse code from finetune_sentiment_classifier with different label column
    return finetune_sentiment_classifier(
        news_df,
        text_column=text_column,
        label_column=label_column,
        model_name=model_name,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size
    )

def finetune_language_model(
    news_df,
    text_column='content',
    model_name='indolem/indobert-base-uncased',
    output_dir='./models/language',
    epochs=3,
    batch_size=8
):
    """
    Fine-tune a language model on Indonesian content.
    
    Args:
        news_df: DataFrame with news articles
        text_column: Column containing text
        model_name: Base model to fine-tune
        output_dir: Output directory for saving model
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Dictionary with training info and model path
    """
    if news_df.empty or text_column not in news_df.columns:
        raise ValueError("DataFrame is empty or missing required column")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    full_dataset = create_training_dataset(news_df, text_column)
    
    # Split dataset
    train_test = full_dataset.train_test_split(test_size=0.2)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = train_test.map(tokenize_function, batched=True)
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )
    
    # Train model
    train_result = trainer.train()
    
    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Evaluation
    eval_result = trainer.evaluate()
    
    # Save training info
    training_info = {
        'model_name': model_name,
        'base_model': model_name,
        'fine_tuned_task': 'language_modeling',
        'dataset_size': len(full_dataset),
        'train_size': len(tokenized_datasets["train"]),
        'test_size': len(tokenized_datasets["test"]),
        'training_loss': train_result.training_loss,
        'evaluation_loss': eval_result['eval_loss'],
        'epochs': epochs,
        'date_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_path': output_dir
    }
    
    # Save training info
    with open(os.path.join(output_dir, 'training_info.json'), 'w') as f:
        json.dump(training_info, f)
    
    return training_info

def analyze_with_finetuned_model(
    text,
    model_path,
    task_type='sentiment'
):
    """
    Analyze text using a fine-tuned model.
    
    Args:
        text: Text to analyze
        model_path: Path to fine-tuned model
        task_type: Type of analysis (sentiment or topic)
        
    Returns:
        Dictionary with analysis results
    """
    if not os.path.exists(model_path):
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Load label mapping
    try:
        with open(os.path.join(model_path, 'label_mapping.json'), 'r') as f:
            label_mapping = json.load(f)
            id2label = label_mapping['id2label']
    except:
        # Use model's mapping if file not found
        id2label = model.config.id2label
    
    # Preprocess text
    processed_text = preprocess_indonesian_text(text)
    
    # Tokenize
    inputs = tokenizer(processed_text, return_tensors="pt", padding=True, truncation=True)
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = outputs.logits
        predicted_class_id = predictions.argmax().item()
    
    # Convert ID to label
    predicted_label = id2label[str(predicted_class_id)]
    
    # Get probabilities
    probs = torch.nn.functional.softmax(predictions, dim=-1)
    confidence = probs[0][predicted_class_id].item()
    
    # Prepare results
    result = {
        'text': text,
        'prediction': predicted_label,
        'confidence': confidence,
        'probabilities': {id2label[str(i)]: probs[0][i].item() for i in range(len(id2label))}
    }
    
    return result