import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
# nltk.download('twitter_samples')
# nltk.download('averaged_perceptron_tagger')
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import streamlit as st
import re
import nltk
from nltk import classify
from nltk import NaiveBayesClassifier
import base64
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics
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
#import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.svm import LinearSVC
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from collections import Counter
from sklearn.metrics import f1_score
import collections
from sklearn.pipeline import Pipeline

def lower_case(sanitize):
    data_lowercase = []
    data_details_lowercase = sanitize.str.lower()
    return data_details_lowercase

def url(sanitize):
    url_all = []
    for word in sanitize:
        data_details_url = re.sub(r"http\S+", "", word)
        url_all.append(data_details_url)
    return url_all


def punctuation(sanitize):
    data_punct = []
    for i in sanitize:
        data_details_punct = re.sub(r'[^\w\s]', '', i)
        data_punct.append(data_details_punct)
    return data_punct


def token(sanitize):
    data_token = []
    for i in sanitize:
        tokenize = nltk.word_tokenize(i)
        data_token.append(tokenize)
    return data_token


def stopWord(sanitize):
    stopword_data = []
    STOPWORDS = set(stopwords.words('english'))
    for k in sanitize:
        sani_stopword = []
        for i in k:
            if i not in STOPWORDS:
                sani_stopword.append(i)
        stopword_data.append(sani_stopword)
    return stopword_data

def strip_html_tags(text):
    ren_htmltags = []
    for i in text:
        soup = BeautifulSoup(i, "html.parser")
        stripped_text = soup.get_text()
        ren_htmltags.append(stripped_text)
    return ren_htmltags

def join_tokens(remove_stopwords):
    join_token = []
    for i in remove_stopwords:
        join_words = ' '.join(i)
        join_token.append(join_words)
    return join_token

def join_positive(positive):
    positive_join = []
    for i in positive:
        print(i)
        join_data = ' '.join(i)
        positive_join.append(join_data)
    return positive_join

if __name__ == "__main__":

    data_lowercase = []
    # url_all = []
    # data_punct = []
    # data_token = []
    # stopword_data = []
    # ren_htmltags = []
    # sani_stopword = []
    # join_token = []

    # getting dataset
    sentiment_new = pd.read_csv(r'prepd_data.csv', error_bad_lines=False, sep=',')
    review = sentiment_new['review']
    target = sentiment_new['sentiment_label']

    #pre processing
    lwr_case_positive = lower_case(review)
    review_positive_remhtml = strip_html_tags(lwr_case_positive)
    remove_punct_positive = punctuation(review_positive_remhtml)
    remove_url = url(remove_punct_positive)
    tweet_tokens = token(remove_url)
    remove_stopwords = stopWord(tweet_tokens)
    join_stopwords = join_tokens(remove_stopwords)

    # spliting test and train dataset
    X_train, X_test, y_train, y_test = train_test_split(join_stopwords, target, test_size=0.3, random_state=1)
    train_data_re = X_train
    test_data_re = X_test
    train_target_re = y_train
    test_target = y_test

    # twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
    # print(twenty_train.data)

    # review_train = X_train_counts[:10]
    # train_data = train_data_re.reshape(-1, 1)
    # print(review_train.shape)
    # review_test = X_train_counts[10:]
    # test_data = test_data_re.reshape(-1, 1)
    # y_train = target[:10]
    # train_target = train_target_re.reshape(-1, 1)
    # print(y_train.shape)
    # y_test = target[10:]

    # # bag of words
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_data_re)
    train_data_re = X_train_counts.reshape(-1, 1)
    X_test_counts = count_vect.transform(test_data_re)
    test_data_re = X_test_counts.reshape(-1, 1)
    #
    # #tfidf
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(train_data_re)

    #
    # tfidf_transformer_test = TfidfTransformer()
    X_test_tfidf = tfidf_transformer.fit_transform(test_data_re)
    #
    # # scale standard
    sc = StandardScaler(with_mean=False)
    sc.fit(X_train_tfidf)
    X_train_std = sc.transform(X_train_tfidf)
    X_test_std = sc.transform(X_test_tfidf)



    # clf = MultinomialNB().fit(X_train_counts, train_target)
    clf = MultinomialNB()
    # mnb_clf = Pipeline([
    #     ('vect', CountVectorizer()),
    #     ('tfidf', TfidfTransformer()),
    #     ('scale', StandardScaler(with_mean=False)),
    #     ('mnb_clf', MultinomialNB()),
    # ])
    # text_clf = Pipeline([('vect', CountVectorizer()),
    #                      ('tfidf', TfidfTransformer()),
    #                      ('clf', MultinomialNB()),])
    text_clf = clf.fit(X_train_std, train_target)
    print(text_clf)
    predicted = text_clf.predict(test_data)
    # y_predicted_labels = le.inverse_transform(predicted)
    # print(y_predicted_labels)
    data = {'Review': test_data, 'labels': predicted}
    df = pd.DataFrame(data)
    print(df.tail(10))
    print(type(test_data))
    print(test_data.shape)
    print(type(predicted))
    classification  = classification_report(test_target,predicted,target_names=['Positive','Negative'])
    print(classification)
    print(np.mean(predicted == test_target))
    print(metrics.accuracy_score(test_target, predicted))
    #
    # data = {'Reviews': ['I ordered just once from TerribleCo, they screwed up, never used the app again.',
    #                     'it is beautiful', 'I am angry', 'i am not interested in this flim', 'it worth watching'],
    #         'Movies': ['wizard of oz', 'evil dead', 'alien', 'Zombie', 'wall street']}