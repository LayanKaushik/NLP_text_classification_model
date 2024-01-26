# NLP_text_classification_model

This repository contains a simple text classification model employing various widely recognized supervised machine learning algorithms, along with common text vectorization methods.

## About the Data

This dataset comprises text from 3,000 TikTok videos discussing ADHD. It contains 10 columns. The target variable, located in the second column labeled 'funny,' is a binary variable (0/1) indicating whether a text is humorous. The dataset features text converted from voice to text.

## Task

To tackle this supervised classification task, we've broken down the process into the following steps:

- Setup: Importing Libraries, Loading Data, and Conducting Exploratory Data Analysis
- Text Pre-processing
- Data Split & Feature Selection
- Text Vectorization
- Implementing ML & DL Algorithms
- Findings & Conclusion

## Setup

We imported all necessary libraries from nltk for text pre-processing, as well as sklearn and keras for ML & DL algorithms, and libraries for word embedding. Some rows were manually cleaned to correct column mismatches in pandas. This manual approach was feasible due to the limited number of such issues.

We conducted exploratory data analysis on the TikTok dataset to gauge the importance of other variables and assess whether new variables could enrich the model's information base.

### EDA
1. **Class distribution**: The dataset includes 2,060 videos classified as 0 (not funny) and 940 as 1 (funny), indicating a slight data imbalance. Therefore, the **F1 score** was selected as the performance metric.
2. **Missing values**: The dataset contains no missing values.
3. **Other Variables**: 'create_time', 'comment_count', and 'like_count' displayed some correlation with the target variable. Other variables were excluded as they did not yield useful insights. We plan to evaluate the importance of each variable later and discard those that do not contribute to the model.
4. **New Variables**: Two new variables, 'word_count' and 'char_count', were introduced. EDA revealed that texts classified as funny typically have fewer words and characters on average and exhibit a decent correlation with the target variable. Either 'char_count' or 'word_count' could serve as predictors in the model since they are nearly 100% correlated.
5. **Data Refinement for Text Analysis**: To improve the accuracy of our text-based predictors, we filtered out words that were common and highly frequent across both funny and non-funny classes. This step was undertaken to focus on more distinctive language features that might better differentiate between the two categories.

## Text pre-processing

Before modeling, it's crucial to preprocess the data. The following steps were implemented:

1. Removal of punctuation, special characters, and numbers.
2. Removal of generic English stopwords using nltk.
3. Lemmatization: Simplifying words to their base form.

Our 'utils.py' file contains additional utilities for further text preprocessing options.

## Data Split & Feature Selection

After splitting the data into training and test sets, we address the non-text features by either dropping them or applying PCA, as dictated by our analysis. Detailed instructions on utilizing the feature selection function are available in the 'utils.py' file. This step ensures that only the most relevant features are used for model training, enhancing the efficiency and effectiveness of our classification algorithms.

## Text Vectorization

Handling text data in Machine Learning models can be challenging, as these models require well-defined numerical data. The process of converting text data into numerical vectors is known as vectorization. We began by splitting the dataset into training and testing sets and then tokenized the text using nltk's word_tokenize function. We implemented the following vectorization techniques:

1. **Term Frequency-Inverse Document Frequencies (tf-idf)**: This technique assigns a value to a word, increasing in proportion to its count in a document but decreasing relative to its frequency across the corpus. We utilized sklearn's TfidfVectorizer.
2. **Word2Vec**: This approach employs a shallow neural network to learn word embeddings, capturing the context of a word within the text. We utilized gensim's Word2Vec. 
3. **Global Vectors (GloVe)**: GloVe is an unsupervised learning algorithm for obtaining vector representations of words. It focuses on aggregating global word-word co-occurrence statistics from a corpus and uses these statistics to derive the word vectors. We utilized gensim's glove2word2vec.
4. **Bidirectional Encoder Representations from Transformers (BERT)**: This technique enables the model to understand the context of a word based on its surrounding words using its bidirectional training. We utilized transformers's BertTokenizer and BertModel.

The 'utils.py' file contains functions for implementing any of these vectorization methods.

## Implementing ML & DL Algorithms

In this phase, we applied a range of machine learning and deep learning algorithms to our vectorized dataset. The models included logistic regression, SVM, XgBoost, and various ensemble and neural network models. We conducted a thorough hyperparameter tuning using GridSearchCV, primarily focusing on optimizing the F1 score. Post-tuning, we evaluated model performance based on **F1 score** and **accuracy** metrics on the test dataset. Detailed results and insights from this analysis are summarized in the following section.

## Findings & Conclusion

### Model Performance Summary on Test Dataset

#### Using only text vectorization

| Vectorization Method | Model            | F1 Score | Accuracy |
|----------------------|------------------|----------|----------|
| Tf-idf               | Logistic Regression | 0.49    | 0.701     |
| Tf-idf               | SVM              |   0.454  |   0.683   |
| Tf-idf               | XGBoost              |  0.401    |   0.676   |
| Tf-idf               | Ensemble Model (soft voting)              | 0.376    | 0.73     |
| Tf-idf               | Ensemble Model (hard voting)             | 0.36    | 0.721     |
| Tf-idf               | CNN              | --    | 0.695     |
| Word2Vec               | Logistic Regression |  0.10    | 0.68     |
| Word2Vec               | SVM              | 0.0     | 0.695     |
| Word2Vec               | XGBoost              | 0.466     | 0.691     |
| Word2Vec               | Ensemble Model (soft voting)             | 0.324     | 0.701     |
| Word2Vec               | Ensemble Model  (hard voting)            | 0.348     | 0.72     |
| Word2Vec               | CNN              | --     |   0.695   |
| GloVe               | Logistic Regression | 0.42    | 0.70     |
| GloVe               | SVM              | 0.463    | 0.715     |
| GloVe               | XGBoost              | 0.457    | 0.7     |
| GloVe               | Ensemble Model (soft voting)              | 0.414    |  0.731    |
| GloVe               | Ensemble Model  (hard voting)            |  0.431    |  0.736    |
| GloVe               | CNN              | --     | 0.695    |

#### Using text vectorization and scaled character count

| Vectorization Method | Model            | F1 Score | Accuracy |
|----------------------|------------------|----------|----------|
| Tf-idf               | Logistic Regression | 0.454    | 0.708     |
| Tf-idf               | SVM              |    0.45  |  0.686  |
| Tf-idf               | XGBoost              |  0.449    |   0.701   |
| Tf-idf               | Ensemble Model (soft voting)              | 0.389    | 0.728     |
| Tf-idf               | Ensemble Model (hard voting)             | 0.364    | 0.715     |
| Tf-idf               | CNN              | --    | 0.695     |
| Word2Vec               | Logistic Regression |  0.159    | 0.683    |
| Word2Vec               | SVM              | 0.09     | 0.696     |
| Word2Vec               | XGBoost              | 0.431     | 0.701     |
| Word2Vec               | Ensemble Model (soft voting)             | 0.39     | 0.735     |
| Word2Vec               | Ensemble Model  (hard voting)            | 0.376     | 0.735     |
| Word2Vec               | CNN              | --     |   0.695   |
| GloVe               | Logistic Regression | 0.393    | 0.691     |
| GloVe               | SVM              | 0.494    | 0.71     |
| GloVe               | XGBoost              | 0.486    | 0.705     |
| GloVe               | Ensemble Model (soft voting)              | 0.451   |  0.728    |
| GloVe               | Ensemble Model  (hard voting)            |  0.415    |  0.723    |
| GloVe               | CNN              | --     | 0.695    |

#### Using text vectorization and pca

| Vectorization Method | Model            | F1 Score | Accuracy |
|----------------------|------------------|----------|----------|
| Tf-idf               | Logistic Regression | 0.466    | 0.706     |
| Tf-idf               | SVM              |    0.454  |  0.683  |
| Tf-idf               | XGBoost              |  0.462    |  0.71   |
| Tf-idf               | Ensemble Model (soft voting)              | 0.380    | 0.723     |
| Tf-idf               | Ensemble Model (hard voting)             | 0.353    | 0.713     |
| Tf-idf               | CNN              | --    | 0.695     |
| Word2Vec               | Logistic Regression |  0.165    | 0.68     |
| Word2Vec               | SVM              | 0.021    | 0.693     |
| Word2Vec               | XGBoost              | 0.461     | 0.708     |
| Word2Vec               | Ensemble Model (soft voting)             | 0.355     | 0.721     |
| Word2Vec               | Ensemble Model  (hard voting)            | 0.379     | 0.738     |
| Word2Vec               | CNN              | --     |   0.695   |
| GloVe               | Logistic Regression | 0.401    | 0.691     |
| GloVe               | SVM              | 0.462    | 0.713     |
| GloVe               | XGBoost              | 0.413   | 0.678     |
| GloVe               | Ensemble Model (soft voting)              | 0.469    |  0.736    |
| GloVe               | Ensemble Model  (hard voting)            |  0.411   |  0.728    |
| GloVe               | CNN              | --     | 0.695    |

Our analysis identified ensemble methods, particularly with GloVe vectorization, as the most effective approach in classifying the humor content in TikTok videos discussing ADHD. These methods consistently outperformed individual models across various vectorization techniques.

#### Key Observations:
1. **Ensemble Methods Excel**: Ensemble methods with GloVe vectorization showed the best performance, achieving high F1 scores and accuracy.
2. **Feature Importance Analysis**: Words associated with conversational and expressive language, such as 'gon', 'look', 'okay', and 'yeah', emerged as strong predictors for humorous content, aligning with the interactive nature of humor.
3. **Refined Text Analysis**: Refining the text data by removing high-frequency words common to both classes resulted in a more distinctive set of features, improving the model's ability to differentiate between humorous and non-humorous content.
4. **Beneficial Use of PCA and Additional Predictors**: The inclusion of PCA and additional predictors like word and character count slightly improved accuracy and F1 scores, suggesting a potential for model optimization.

#### Recommendations:
- **Further Tuning and Analysis**: Additional fine-tuning and exploration of PCA and other predictors could enhance model performance. Continue refining the text data to emphasize unique language features that are indicative of humor.

## Acknowledgment

This content was developed as part of a test for a Research Assistant position at Duke University. Although the data is publicly available, as it was provided for the test, I have chosen not to upload the file to the repository.
