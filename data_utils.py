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

def clean_review(text):
    text = re.sub(r'/><br', '', text)  # Supprime les balises HTML
    text = re.sub(r'review|following|expresses|sentiment\?', '', text)  # Supprime les termes indésirables
    return text

def get_top_n_words(word_counter, n=10):
    return word_counter.most_common(n)

def create_word_counter(dataset):
    word_counter = Counter()
    for review in dataset['inputs_pretokenized']:
        cleaned_review = clean_review(review)
        tokens = [word for word in cleaned_review.split() if word.lower() not in stop_words]
        word_counter.update(tokens)
    return word_counter

def tokenize_function(examples):
    return tokenizer(examples['inputs_pretokenized'], truncation=True, padding=True, max_length=256)

def label_to_int(example):
    return {'label': 1 if example['targets_pretokenized'] == 'positive' else 0}

