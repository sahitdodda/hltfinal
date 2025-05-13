import os
import torch
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from pathlib import Path
import re

# Directories
# Option 1: Local Colab storage (temporary, reset on session end)
CORPUS_DIR = "/content/corpora"  # Where your corpus creation code saves tokencorpus_*.txt
OUTPUT_DIR = "/content/histbert_models"  # Where to save models

# Option 2: Google Drive (uncomment to use, ensure corpora is in Drive)
# CORPUS_DIR = "/content/drive/My Drive/histbert/corpora"
# OUTPUT_DIR = "/content/drive/My Drive/histbert/histbert_models"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load BERT tokenizer and model
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForMaskedLM.from_pretrained(MODEL_NAME)

# Function to load and preprocess a tokencorpus_*.txt file
def load_corpus_file(file_path):
    texts = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("tokens:"):
                    text = line[len("tokens: "):].strip()
                    if text:
                        texts.append(text)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return texts

# Function to create a Hugging Face Dataset from texts
def create_dataset(texts):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=128,  # Memory-efficient for Colab
            return_special_tokens_mask=True
        )

    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset

# Main function to fine-tune HistBERT for a time slice
def fine_tune_histbert(corpus_file, time_slice):
    print(f"\n=== Fine-Tuning HistBERT for {time_slice} ===")

    # Load and preprocess corpus
    texts = load_corpus_file(corpus_file)
    if not texts:
        print(f"No valid texts found in {corpus_file}")
        return

    # Create dataset
    dataset = create_dataset(texts)

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Training arguments for Colab
    output_model_dir = os.path.join(OUTPUT_DIR, f"histbert_{time_slice}")
    training_args = TrainingArguments(
        output_dir=output_model_dir,
        overwrite_output_dir=True,
        num_train_epochs=2,  # Suitable for small corpus
        per_device_train_batch_size=4,  # Memory-efficient
        save_steps=200,  # Frequent saves for Colab stability
        save_total_limit=2,
        logging_dir=os.path.join(output_model_dir, "logs"),
        logging_steps=100,
        learning_rate=2e-5,
        fp16=True,  # Mixed precision for GPU
        report_to="none",  # Disable external logging
        max_steps=500,  # Limit steps to avoid timeout
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Fine-tune
    print(f"Training on {len(dataset)} examples...")
    trainer.train()

    # Save model
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"Saved HistBERT model to {output_model_dir}")

# Process all tokencorpus_*.txt files
def main():
    corpus_files = sorted(Path(CORPUS_DIR).glob("tokencorpus_*.txt"))
    if not corpus_files:
        print(f"No corpus files found in {CORPUS_DIR}")
        return

    for corpus_file in corpus_files:
        time_slice = re.search(r"tokencorpus_(\d{4}-\d{4})\.txt", corpus_file.name)
        if time_slice:
            time_slice = time_slice.group(1)
            fine_tune_histbert(corpus_file, time_slice)
        else:
            print(f"Skipping {corpus_file}: Invalid filename format")

if __name__ == "__main__":
    main()
