import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
# nltk.download('twitter_samples')
# nltk.download('averaged_perceptron_tagger')
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import re
import nltk
from nltk import classify
from nltk import NaiveBayesClassifier
import streamlit as st
import base64
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import re, string, random
from bs4 import BeautifulSoup
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def lower_case(sanitize):
    data_details_lowercase = sanitize.str.lower()
    return data_details_lowercase


def url(sanitize):
    for word in sanitize:
        data_details_url = re.sub(r"http\S+", "", word)
        url_all.append(data_details_url)
    #         data_details_url = re.compile(r'https?://\S+|www\.\S+')
    #         data_details_url = data_details_url.sub(r'', word)
    return url_all


def punctuation(sanitize):
    for i in sanitize:
        data_details_punct = re.sub(r'[^\w\s]', '', i)
        data_punct.append(data_details_punct)
        # data_details_punct = i.str.replace('[^\w\s]','')
    return data_punct


def token(sanitize):
    for i in sanitize:
        tokenize = nltk.word_tokenize(i)
        data_token.append(tokenize)
        # print(data_token)
    return data_token


def stopWord(sanitize):
    # san = [['this','is','rose','smells','good'],['this','is','blue','moon']]
    STOPWORDS = set(stopwords.words('english'))
    for k in sanitize:
        # print(k)
        sani_stopword = []
        for i in k:
            if i not in STOPWORDS:
                # print(i)
                sani_stopword.append(i)
                # print(sani_stopword)
        # print(sani_stopword)
        stopword_data.append(sani_stopword)
        # print(stopword_data)

    return stopword_data

def strip_html_tags(text):
    for i in text:
        # print(i)
        soup = BeautifulSoup(i, "html.parser")
        # print(soup)
        stripped_text = soup.get_text()
        ren_htmltags.append(stripped_text)
        # print(ren_htmltags)
    return ren_htmltags


if __name__ == "__main__":

    data_lowercase = []
    url_all = []
    data_punct = []
    data_token = []
    stopword_data = []
    ren_htmltags = []
    sani_stopword = []
    join_token = []

    le = preprocessing.LabelEncoder()

    sentiment_new = pd.read_csv(r'IMDB_dataset.csv', error_bad_lines=False, sep=',')
    # sentiment_new = sentiment_new.head(20)
    sentiment_new['sentiment_words'] = sentiment_new.sentiment.apply(lambda x: 'positive' if x == 1 else 'negative')
    review = sentiment_new['review']
    target = sentiment_new['sentiment']
    lwr_case_positive = lower_case(review)
    review_positive_remhtml = strip_html_tags(lwr_case_positive)
    remove_punct_positive = punctuation(review_positive_remhtml)
    remove_url = url(remove_punct_positive)
    # stop_words = stopwords.words('english')
    tweet_tokens = token(remove_url)
    remove_stopwords = stopWord(tweet_tokens)
    print(remove_stopwords)
    # san = [['this', 'is', 'rose', 'smells', 'good'], ['this', 'is', 'blue', 'moon']]
    for i in remove_stopwords:
        join_words = ' '.join(i)
        join_token.append(join_words)
    # print(join_token)
        # break
    # print("join")
    # print(join_token)
    # y = target.apply(le.fit_transform)
    X_train, X_test, y_train, y_test = train_test_split(join_token, target, random_state=1)
    train_data = X_train
    test_data = X_test
    train_target = y_train
    test_target = y_test


    # twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
    # print(twenty_train.data)

    # review_train = X_train_counts[:10]
    # review_train = review_train.reshape(-1, 1)
    # print(review_train.shape)
    # review_test = X_train_counts[10:]
    # review_test = review_test.reshape(-1, 1)
    # y_train = target[:10]
    # # y_train = y_train.reshape(-1, 1)
    # print(y_train.shape)
    # y_test = target[10:]


    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_data)
    X_test_counts = count_vect.transform(test_data)
    # print(X_train_counts)

    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # # print(X_train_tfidf)



    # clf = MultinomialNB().fit(X_train_counts, train_target)
    clf = MultinomialNB()
    # text_clf = Pipeline([('vect', CountVectorizer()),
    #                      ('tfidf', TfidfTransformer()),
    #                      ('clf', MultinomialNB()),])
    text_clf = clf.fit(X_train_counts, train_target)
    print(text_clf)
    predicted = text_clf.predict(X_test_counts)
    # y_predicted_labels = le.inverse_transform(predicted)
    # print(y_predicted_labels)
    data = {'Review': test_data, 'labels': predicted}
    df = pd.DataFrame(data)
    print(df.tail(10))
    print(type(test_data))
    print(X_test_counts.shape)
    print(type(predicted))
    classification  = classification_report(test_target,predicted,target_names=['Positive','Negative'])
    print(classification)
    print(np.mean(predicted == test_target))
    print(metrics.accuracy_score(test_target, predicted))

    data = {'Reviews': ['I ordered just once from TerribleCo, they screwed up, never used the app again.',
                        'it is beautiful', 'I am angry', 'i am not interested in this flim', 'it worth watching'],
            'Movies': ['wizard of oz', 'evil dead', 'alien', 'Zombie', 'wall street']}