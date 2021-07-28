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
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from collections import Counter
import collections

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import re
import nltk
from nltk import classify
from sklearn.metrics import f1_score
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
    print(target)
    print(review.head(10))
    lwr_case_positive = lower_case(review)
    review_positive_remhtml = strip_html_tags(lwr_case_positive)
    remove_punct_positive = punctuation(review_positive_remhtml)
    remove_url = url(remove_punct_positive)
    # stop_words = stopwords.words('english')
    tweet_tokens = token(remove_url)
    remove_stopwords = stopWord(tweet_tokens)

    #print(tweet_tokens)
    # san = [['this', 'is', 'rose', 'smells', 'good'], ['this', 'is', 'blue', 'moon']]
    for i in remove_stopwords:
        join_words = ' '.join(i)
        join_token.append(join_words)

        # break
    # print("join")
    # print(join_token)
    # y = target.apply(le.fit_transform)
    X_train, X_test, y_train, y_test = train_test_split(join_token, target, test_size=0.3, random_state=1)
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
    print('BOW_cv_train:', X_train_counts.shape)
    print('BOW_cv_test:', X_test_counts.shape)


    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    tfidf_transformer_test = TfidfTransformer()
    X_test_tfidf = tfidf_transformer_test.fit_transform(X_test_counts)

    sc = StandardScaler(with_mean=False)
    sc.fit(X_train_tfidf)
    X_train_std = sc.transform(X_train_tfidf)
    X_test_std = sc.transform(X_test_tfidf)

    # Instantiate the Support Vector Classifier (SVC)
    from sklearn.svm import LinearSVC

    svc = LinearSVC(dual=False)

    # Fit the model
    text_svc = svc.fit(X_train_std, train_target)

    #print(text_svc)
    # Make the predictions
    y_predict = svc.predict(X_test_std)

    data = {'Review': test_data, 'labels': y_predict}
    df = pd.DataFrame(data)
    positive_data = df.loc[df['labels'] == 1]
    positive_review = positive_data['Review'].tolist()
    positive_words = ' '.join(positive_review)
    print(positive_words)
    negative_data = df.loc[df['labels'] == 0]
    negative_review = negative_data['Review'].tolist()
    negative_words = ' '.join(negative_review)
    print(negative_words)

    print(df.tail(10))
    print(type(test_data))
    print(X_test_counts.shape)
    print(type(y_predict))
    classification  = classification_report(test_target,y_predict,target_names=['Positive','Negative'])
    print(classification)
    print(np.mean(y_predict == test_target))

    # Measure the performance
    print("Accuracy score %.3f" % metrics.accuracy_score(test_target, y_predict))

    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(test_target, y_predict))

    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(y_test, y_predict))

    # F1 score
    print("F1 score:", f1_score(y_test, y_predict, average='macro'))


    target_cnt = Counter(target)

    plt.figure(figsize=(16,8))
    plt.bar(target_cnt.keys(), target_cnt.values())
    plt.title("Dataset labels distribuition")
    plt.show()

    #Confusion matrix

    conf = confusion_matrix(test_target, y_predict)
    print(conf)

    cm = pd.DataFrame(
        conf, index = [i for i in ['0', '1']],
        columns = [i for i in ['0', '1']]
    )

    plt.figure(figsize = (12,7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()

    plt.figure(figsize=(20, 10))
    WC = WordCloud(width=1600, height=800, background_color="rgba(255, 255, 255, 0)").generate(positive_words)
    plt.imshow(WC)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    plt.savefig('wordcloud.png', facecolor='k', bbox_inches='tight')

    plt.figure(figsize=(20, 10))
    WC = WordCloud(width=1600, height=800, background_color="rgba(255, 255, 255, 0)").generate(negative_words)
    plt.imshow(WC)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    plt.savefig('wordcloud.png', facecolor='k', bbox_inches='tight')


