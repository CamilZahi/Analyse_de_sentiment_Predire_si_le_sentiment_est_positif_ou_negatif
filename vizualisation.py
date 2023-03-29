# Importation des librairies

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

def compute_average_word_count(train_data, test_data, unsupervised_data):
  train_word_count = np.mean([len(review.split()) for review in train_data['inputs_pretokenized']])
  test_word_count = np.mean([len(review.split()) for review in test_data['inputs_pretokenized']])
  unsupervised_word_count = np.mean([len(review.split()) for review in unsupervised_data['inputs_pretokenized']])
  return train_word_count, test_word_count, unsupervised_word_count

def plot_average_word_count(train_count, test_count, unsupervised_count):
  labels = ['Train', 'Test', 'Unsupervised']
  avg_word_counts = [train_count, test_count, unsupervised_count]

  x = np.arange(len(labels))
  width = 0.5

  fig, ax = plt.subplots()
  rects = ax.bar(x, avg_word_counts, width)

  ax.set_ylabel('Nombre moyen de mots par phrase')
  ax.set_title("Nombre moyen de mots par phrase dans chaque ensemble")
  ax.set_xticks(x)
  ax.set_xticklabels(labels)

  def autolabel(rects):
      for rect in rects:
          height = rect.get_height()
          ax.annotate('{:.2f}'.format(height),
                      xy=(rect.get_x() + rect.get_width() / 2, height),
                      xytext=(0, 3),
                      textcoords="offset points",
                      ha='center', va='bottom')

  autolabel(rects)

  fig.tight_layout()
  plt.show()

def plot_sentence_lengths_histogram(subsets, labels, num_bins=50, colors=None):
  fig, ax = plt.subplots()
  
  if colors is None:
      colors = ['green', 'red', 'skyblue']
  
  for subset, label, color in zip(subsets, labels, colors):
      sentence_lengths = [len(review.split()) for review in subset['inputs_pretokenized']]
      ax.hist(sentence_lengths, bins=num_bins, alpha=0.5, label=label, edgecolor='black', color=color, density=True) # Ajouter density=True ici

  ax.set_title("Distribution des longueurs de phrases")
  ax.set_xlabel("Nombre de mots")
  ax.set_ylabel("Proportion de phrases")
  ax.legend()

  plt.show()

def generate_wordcloud(word_counter):
  wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis')
  wordcloud.generate_from_frequencies(word_counter)
  return wordcloud

def display_wordcloud(wordcloud):
  plt.figure(figsize=(12, 6))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis('off')
  plt.show()

def count_emoticons_and_punctuation(subset):
    emoticons = r'[:;=]-?[()DOP]|\(?\)C'
    punctuation = r'[.,!?;-]'
    emoticons_count = 0
    punctuation_count = 0
    for review in subset['inputs_pretokenized']:
        emoticons_count += len(re.findall(emoticons, review))
        punctuation_count += len(re.findall(punctuation, review))
    return emoticons_count, punctuation_count

def plot_emoticons_punctuation_separate(train_emoticons, train_punctuation, test_emoticons, test_punctuation, unsupervised_emoticons, unsupervised_punctuation, train_size, test_size, unsupervised_size):
    labels = ['Train', 'Test', 'Unsupervised']
    
    emoticons_data = [train_emoticons / train_size, test_emoticons / test_size, unsupervised_emoticons / unsupervised_size]
    punctuation_data = [train_punctuation / train_size, test_punctuation / test_size, unsupervised_punctuation / unsupervised_size]

    x = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    # Émoticônes
    ax1.bar(x, emoticons_data, color='lightgreen', width=0.4)
    ax1.set_ylabel('Proportion')
    ax1.set_title("Proportion d'émoticônes")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    # Ponctuations
    ax2.bar(x, punctuation_data, color='salmon', width=0.4)
    ax2.set_ylabel('Proportion')
    ax2.set_title("Proportion de ponctuations")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)

    plt.show()

