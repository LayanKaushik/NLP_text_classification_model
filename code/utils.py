# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
from collections import Counter

#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

#for feature-selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer

#for word embedding
import gensim
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

#for BERT Vectorization
from transformers import BertTokenizer, BertModel
import torch

def print_separator(title):
    print("\n" + "=" * 50)
    print(f"{title}".center(50))
    print("=" * 50)

def clean_text(text):
    """
    Convert text to lowercase, remove punctuation and special characters.
    :param text: str - The text to be cleaned.
    :return: str - The cleaned text.
    """
    text = text.lower().strip()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def remove_stopwords(text):
    """
    Remove English stopwords from the text.
    :param text: str - The text to be processed.
    :return: str - Text without stopwords.
    """
    stop_words = set(stopwords.words('english'))
    return ' '.join(word for word in text.split() if word not in stop_words)

def stem_text(text, stemmer_type='porter'):
    """
    Stem the words in the text.
    :param text: str - The text to be stemmed.
    :param stemmer_type: str - Type of stemmer to use ('porter', 'lancaster', 'snowball').
    :return: str - The stemmed text.
    """
    if stemmer_type == 'porter':
        stemmer = PorterStemmer()
    elif stemmer_type == 'lancaster':
        stemmer = LancasterStemmer()
    elif stemmer_type == 'snowball':
        stemmer = SnowballStemmer('english')
    else:
        raise ValueError("Invalid stemmer_type. Choose 'porter', 'lancaster', or 'snowball'")
    
    return ' '.join([stemmer.stem(word) for word in text.split()])

def get_wordnet_pos(word):
    """
    Map POS tag to a format recognized by lemmatize() function.
    :param word: str - A single word.
    :return: str - Corresponding WordNet POS tag.
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    """
    Lemmatize the text.
    :param text: str - The text to be lemmatized.
    :return: str - The lemmatized text.
    """
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(text)])

def preprocess_text(text, remove_sw=True, stemming=False, lemmatization=False, stemmer_type='porter'):
    """
    Preprocess the text by cleaning, removing stopwords, and applying stemming or lemmatization.
    :param text: str - The text to be processed.
    :param remove_sw: bool - Flag to indicate whether to remove stopwords.
    :param stemming: bool - Flag to indicate whether to apply stemming.
    :param lemmatization: bool - Flag to indicate whether to apply lemmatization.
    :param stemmer_type: str - Type of stemmer to use.
    :return: str - The processed text.
    """
    text = clean_text(text)
    if remove_sw:
        text = remove_stopwords(text)
    if stemming:
        text = stem_text(text, stemmer_type=stemmer_type)
    if lemmatization:
        text = lemmatize_text(text)
    return text

def count_words(data, key):
    """
    Counts the frequency of words in a specified column of the DataFrame.
    :param data: The DataFrame containing the data.
    :param key: str - The column name in the DataFrame to count words from.
    :return: obj - A Counter object with word frequencies.
    """
    return Counter([word for sublist in data[key] for word in sublist])

def perform_pca(train_data, test_data, features, n_components=2):
    """
    Fit PCA on the training data and transform both training and test data.
    :param train_data: DataFrame containing the training dataset.
    :param test_data: DataFrame containing the test dataset.
    :param features: List of features for PCA.
    :param n_components: Number of PCA components to keep.
    :return: DataFrames with principal components for both training and test data.
    """
    x_train = train_data.loc[:, features].values
    x_test = test_data.loc[:, features].values
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    pca = PCA(n_components=n_components)
    pca.fit(x_train)

    principalComponents_train = pca.transform(x_train)
    principalComponents_test = pca.transform(x_test)

    columns = [f'PC{i+1}' for i in range(n_components)]
    principalDf_train = pd.DataFrame(data=principalComponents_train, columns=columns)
    principalDf_test = pd.DataFrame(data=principalComponents_test, columns=columns)

    return principalDf_train, principalDf_test

def process_data(train_data, test_data, drop=True, pca_features=None, drop_features=None, n_components=None):
    """
    Process the dataset by either dropping specified features or performing PCA.
    :param train_data: DataFrame containing the training dataset.
    :param test_data: DataFrame containing the test dataset.
    :param drop: Boolean flag to indicate whether to drop features.
    :param pca_features: List of features for PCA.
    :param drop_features: List of features to drop.
    :param n_components: Number of PCA components to keep.
    :return: Processed DataFrames for both training and test data.
    """
    if drop:
        train_data = train_data.drop(columns=drop_features)
        test_data = test_data.drop(columns=drop_features)
        return train_data, test_data
    else:
        pca_df_train, pca_df_test = perform_pca(train_data, test_data, pca_features, n_components)

        # Combining PCA data with the rest of the dataset
        remaining_features = train_data.columns.difference(pca_features)
        combined_train_data = pd.concat([train_data.loc[:, remaining_features].reset_index(drop=True), pca_df_train], axis=1)
        combined_test_data = pd.concat([test_data.loc[:, remaining_features].reset_index(drop=True), pca_df_test], axis=1)

        return combined_train_data, combined_test_data

def tokenize_texts(texts):
    """
    Tokenize a list of texts.
    :param texts: List of strings to tokenize.
    :return: List of tokenized texts.
    """
    return [nltk.word_tokenize(text) for text in texts]

def mean_embedding_vectorizer(words, model, vector_size):
    """
    Calculate the mean embedding for a list of words using a given word embedding model.
    :param words: List of words to vectorize.
    :param model: Word embedding model.
    :param vector_size: Size of the word vectors.
    :return: Mean vector of the words.
    """
    return np.mean([model[word] for word in words if word in model]
                   or [np.zeros(vector_size)], axis=0)
    
def encode_text(text, tokenizer, model):
    """
    Encode a single text using BERT tokenizer and model.
    :param text: Text to encode.
    :param tokenizer: BERT tokenizer.
    :param model: BERT model.
    :return: Encoded vector representation of the text.
    """
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()

def vectorize_tfidf(texts):
    """
    Vectorize texts using the Tf-Idf method.
    :param texts: List of texts to vectorize.
    :return: Tuple of Tf-Idf vectors and the vectorizer.
    """
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    vectors = tfidf_vectorizer.fit_transform(texts)
    return vectors, tfidf_vectorizer
    
def vectorize_word2vec(tokenized_texts, model=None):
    """
    Vectorize tokenized texts using the Word2Vec method.
    :param tokenized_texts: List of tokenized texts.
    :param model: Model used.
    :return: Tuple of Word2Vec vectors and the model.
    """
    if model is None:
        model = Word2Vec(tokenized_texts, min_count=1)

    w2v = {word: model.wv[word] for word in model.wv.index_to_key}
    vector_size = model.wv.vector_size
    vectors = np.array([mean_embedding_vectorizer(text, w2v, vector_size) for text in tokenized_texts])
    return vectors, model

def vectorize_glove(tokenized_texts, glove_input_file=None, word2vec_output_file=None, model=None):
    """
    Vectorize tokenized texts using the GloVe method.
    :param tokenized_texts: List of tokenized texts.
    :param glove_input_file: Path to the GloVe model file.
    :param word2vec_output_file: Path for the output Word2Vec format file.
    :return: Tuple of GloVe vectors and the model.
    """
    if model is None:
        glove2word2vec(glove_input_file, word2vec_output_file)
        model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    vector_size = model.vector_size
    vectors = np.array([mean_embedding_vectorizer(text, model, vector_size) for text in tokenized_texts])
    return vectors, model
    
def vectorize_bert(texts, bert_model_name='bert-base-uncased'):
    """
    Vectorize texts using the BERT method.
    :param texts: List of texts to vectorize.
    :param bert_model_name: Name of the BERT model.
    :return: Tuple of BERT vectors, tokenizer, and model.
    """
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    model = BertModel.from_pretrained(bert_model_name)
    vectors = np.array([encode_text(text, tokenizer, model) for text in texts])
    return vectors, tokenizer, model

def vectorize_text(texts, method='tfidf', glove_input_file=None, word2vec_output_file=None, bert_model_name='bert-base-uncased'):
    """
    Vectorize texts based on the specified method.
    :param texts: List of texts to vectorize.
    :param method: Method for text vectorization ('tfidf', 'word2vec', 'glove', or 'bert').
    :param glove_input_file: Path to the GloVe model file (required for 'glove').
    :param word2vec_output_file: Path for the output Word2Vec format file (required for 'glove').
    :param bert_model_name: Name of the BERT model (required for 'bert').
    :return: Vectorized texts along with model/tokenizer if applicable.
    """
    if method in ['word2vec', 'glove']:
        tokenized_texts = tokenize_texts(texts)

    if method == 'tfidf':
        return vectorize_tfidf(texts)
    elif method == 'word2vec':
        return vectorize_word2vec(tokenized_texts)
    elif method == 'glove':
        return vectorize_glove(tokenized_texts, glove_input_file, word2vec_output_file)
    elif method == 'bert':
        return vectorize_bert(texts, bert_model_name)
    else:
        raise ValueError("Invalid method. Please select 'tfidf', 'word2vec', 'glove', or 'bert'.")