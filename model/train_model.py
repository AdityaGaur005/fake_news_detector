import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
import json, datetime
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
import logging

# ---------------- Logging ---------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Config ---------------- #
CONFIG = {
    'max_length': 256,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 5,
    'warmup_steps': 1000,
    'test_size': 0.15,
    'val_size': 0.10,
    'random_state': 42,
}

# ---------------- Data Loading ---------------- #
def load_and_prepare_data():
    """Load ISOT + WELFake, align labels, merge, shuffle."""
    try:
        true_news = pd.read_csv(r"C:\Users\Aditya Gaur\Downloads\.vscode\fake\True.csv")
        fake_news = pd.read_csv(r"C:\Users\Aditya Gaur\Downloads\.vscode\fake\Fake.csv")
        welfake = pd.read_csv(
            r"C:\Users\Aditya Gaur\Downloads\.vscode\fake\WELFake_Dataset.csv",
            encoding="latin1",
            low_memory=False
        )

        logger.info(f"Loaded ISOT True: {len(true_news)}")
        logger.info(f"Loaded ISOT Fake: {len(fake_news)}")
        logger.info(f"Loaded WELFake:   {len(welfake)}")

    except FileNotFoundError as e:
        logger.error(f"CSV files not found: {e}")
        raise

    # Label mapping
    true_news["label"] = 1
    fake_news["label"] = 0
    true_news = true_news[['title', 'text', 'label']]
    fake_news = fake_news[['title', 'text', 'label']]

    welfake = welfake[['title', 'text', 'label']].dropna(subset=['label'])
    welfake['label'] = welfake['label'].astype(str).str.strip()
    label_map = {'1': 0, '0': 1, 1: 0, 0: 1}  # WELFake convention
    welfake['label'] = welfake['label'].map(label_map)
    welfake = welfake.dropna(subset=['label'])
    welfake['label'] = welfake['label'].astype(int)

    dataset = pd.concat([true_news, fake_news, welfake], ignore_index=True)
    dataset = dataset.drop_duplicates(subset=['title', 'text']).reset_index(drop=True)
    dataset = dataset.sample(frac=1, random_state=CONFIG['random_state']).reset_index(drop=True)

    logger.info(f"Merged dataset size: {dataset.shape}")
    logger.info(f"Real (label=1): {int((dataset['label']==1).sum())}")
    logger.info(f"Fake (label=0): {int((dataset['label']==0).sum())}")
    return dataset

# ---------------- Cleaning ---------------- #
def clean_text_advanced(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_text_data(dataset):
    dataset['title'] = dataset['title'].fillna('')
    dataset['text']  = dataset['text'].fillna('')
    dataset['combined_text'] = (dataset['title'] + ' ') * 2 + '[SEP] ' + dataset['text']
    dataset['clean_text'] = dataset['combined_text'].apply(clean_text_advanced)
    dataset = dataset[dataset['clean_text'].str.len() > 10].reset_index(drop=True)

    logger.info(f"After cleaning: {len(dataset)} articles remain")
    logger.info(f"Avg clean_text length: {dataset['clean_text'].str.len().mean():.0f} chars")
    return dataset

# ---------------- Split ---------------- #
def create_train_val_test_split(texts, labels):
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels,
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=labels
    )
    val_ratio_within_train = CONFIG['val_size'] / (1 - CONFIG['test_size'])
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels,
        test_size=val_ratio_within_train,
        random_state=CONFIG['random_state'],
        stratify=train_val_labels
    )
    logger.info(f"Train size: {len(train_texts)}")
    logger.info(f"Validation size: {len(val_texts)}")
    logger.info(f"Test size: {len(test_texts)}")
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

# ---------------- Tokenization ---------------- #
def tokenize_data(tokenizer, texts, labels):
    encodings = tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=CONFIG['max_length'],
        return_tensors="pt"
    )
    labels_tensor = torch.tensor(list(labels), dtype=torch.long)
    return encodings, labels_tensor

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    def __len__(self):
        return len(self.labels)

# ---------------- Metrics ---------------- #
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}

# ---------------- Main ---------------- #
def main():
    logger.info("🚀 Starting fake news classifier training...")

    dataset = load_and_prepare_data()
    dataset = prepare_text_data(dataset)

    # ⚡ Debug mode: uncomment for quick tests
    # dataset = dataset.sample(2000, random_state=CONFIG['random_state']).reset_index(drop=True)

    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = create_train_val_test_split(
        dataset['clean_text'], dataset['label']
    )

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    train_encodings, train_labels_tensor = tokenize_data(tokenizer, train_texts, train_labels)
    val_encodings, val_labels_tensor     = tokenize_data(tokenizer, val_texts,   val_labels)
    test_encodings, test_labels_tensor   = tokenize_data(tokenizer, test_texts,  test_labels)

    train_dataset = NewsDataset(train_encodings, train_labels_tensor)
    val_dataset   = NewsDataset(val_encodings,   val_labels_tensor)
    test_dataset  = NewsDataset(test_encodings,  test_labels_tensor)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=CONFIG['num_epochs'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        warmup_steps=CONFIG['warmup_steps'],
        weight_decay=0.01,
        learning_rate=CONFIG['learning_rate'],
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="steps",   # ✅ fixed
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to=None,
        dataloader_pin_memory=False,
        fp16=torch.cuda.is_available(),   # ✅ GPU safe
        gradient_accumulation_steps=2,
        lr_scheduler_type="cosine"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    logger.info("⚡ Training started...")
    trainer.train()

    logger.info("📊 Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    for key, value in test_results.items():
        if isinstance(value, (int, float, np.floating)):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")

    logger.info("💾 Saving model and tokenizer...")
    model.save_pretrained('./final_model')
    tokenizer.save_pretrained('./final_model')

    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = test_labels.values
    report = classification_report(y_true, y_pred, target_names=['Fake', 'Real'], digits=4)
    logger.info("Detailed Classification Report:\n" + report)

    # ✅ Save a summary of training results
    results_summary = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "accuracy": float(test_results.get("eval_accuracy", 0)),
        "f1": float(test_results.get("eval_f1", 0)),
        "precision": float(test_results.get("eval_precision", 0)),
        "recall": float(test_results.get("eval_recall", 0))
    }

    with open("training_summary.json", "w") as f:
        json.dump(results_summary, f, indent=4)

    logger.info("Training summary saved to training_summary.json")

    logger.info("✅ Training completed successfully!")
    return trainer, test_results

if __name__ == "__main__":
    trainer, results = main()
