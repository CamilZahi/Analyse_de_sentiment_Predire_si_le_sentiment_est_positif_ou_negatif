from datasets import load_dataset
from datasets import load_metric
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from transformers import Trainer, TrainingArguments
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

metric = load_metric("accuracy")
recall = load_metric("recall")
precision = load_metric("precision")
f1 = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': metric.compute(predictions=predictions, references=labels)['accuracy'],
        'recall': recall.compute(predictions=predictions, references=labels, average='macro')['recall'],
        'precision': precision.compute(predictions=predictions, references=labels, average='macro')['precision'],
        'f1': f1.compute(predictions=predictions, references=labels, average='macro')['f1'],
    }
def get_trainer(model, train_dataset, test_dataset):
  training_args = TrainingArguments(
      output_dir='./results',
      evaluation_strategy='epoch',
      num_train_epochs=1,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=32,
      warmup_steps=500,
      weight_decay=0.1,
      logging_dir='./logs',
      logging_steps=10,
  )
  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset,
      eval_dataset=test_dataset,
      compute_metrics=compute_metrics, 
  )
  return trainer