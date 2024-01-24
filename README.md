# NLP_text_classification_model

This repository contains a simple text classification model employing various widely recognized supervised machine learning algorithms, along with common text vectorization methods.

## About the Data

This dataset comprises text from 3,000 TikTok videos discussing ADHD. It contains 10 columns. The target variable, located in the second column labeled 'funny,' is a binary variable (0/1) indicating whether a text is humorous. The dataset features text converted from voice to text.

## Task

To tackle this supervised classification task, we've broken down the process into the following steps:

- Setup: Importing Libraries, Loading Data, and Conducting Exploratory Data Analysis
- Text Pre-processing
- Text Vectorization
- Implementing ML & DL Algorithms
- Findings

## Setup

We imported all necessary libraries from nltk for text pre-processing, as well as sklearn and keras for ML & DL algorithms, and libraries for word embedding. Some rows were manually cleaned to correct column mismatches in pandas. This manual approach was feasible due to the limited number of such issues.

We conducted exploratory data analysis on the TikTok dataset to gauge the importance of other variables and assess whether new variables could enrich the model's information base.

### EDA
1. **Class distribution**: The dataset includes 2,060 videos classified as 0 (not funny) and 940 as 1 (funny), indicating a slight data imbalance. Therefore, the **F1 score** was selected as the performance metric.
2. **Missing values**: The dataset contains no missing values.
3. **Other Variables**: 'create_time', 'comment_count', and 'like_count' displayed some correlation with the target variable. Other variables were excluded as they did not yield useful insights. We plan to evaluate the importance of each variable later and discard those that do not contribute to the model.
4. **New Variables**: Two new variables, 'word_count' and 'char_count', were introduced. EDA revealed that texts classified as funny typically have fewer words and characters on average and exhibit a decent correlation with the target variable. Either 'char_count' or 'word_count' could serve as predictors in the model since they are nearly 100% correlated.

## Text pre-processing

Before modeling, it's crucial to preprocess the data. The following steps were implemented:

1. Removal of punctuation, special characters, and numbers.
2. Removal of generic English stopwords using nltk.
3. Lemmatization: Simplifying words to their base form.

Our 'utils.py' file contains additional utilities for further text preprocessing options.

## Text Vectorization

Handling text data in Machine Learning models can be challenging, as these models require well-defined numerical data. The process of converting text data into numerical vectors is known as vectorization. We began by splitting the dataset into training and testing sets and then tokenized the text using nltk's word_tokenize function. We implemented the following vectorization techniques:

1. **Term Frequency-Inverse Document Frequencies (tf-idf)**: This technique assigns a value to a word, increasing in proportion to its count in a document but decreasing relative to its frequency across the corpus. We utilized sklearn's TfidfVectorizer.
2. **Word2Vec**: This approach employs a shallow neural network to learn word embeddings, capturing the context of a word within the text. We utilized gensim's Word2Vec. 
3. **Global Vectors (GloVe)**: GloVe is an unsupervised learning algorithm for obtaining vector representations of words. It focuses on aggregating global word-word co-occurrence statistics from a corpus and uses these statistics to derive the word vectors. We utilized gensim's glove2word2vec.
4. **Bidirectional Encoder Representations from Transformers (BERT)**: This technique enables the model to understand the context of a word based on its surrounding words using its bidirectional training. We utilized pytorch's BertTokenizer and BertModel.

The 'utils.py' file contains functions for implementing any of these vectorization methods.

## Implementing ML & DL Algorithms

TODO

## Findings

TODO (Table and word embedding analysis, brief description of the strongest predictors)

## Acknowledgment

This content was developed as part of a test for a Research Assistant position at Duke University. Although the data is publicly available, as it was provided for the test, I have chosen not to upload the file to the repository.
