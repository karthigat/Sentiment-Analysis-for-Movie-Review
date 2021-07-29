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

def lemitization(sanitize):
    data_lem = []
    for j in sanitize:
        lem = [lemiti.lemmatize(i) for i in j]
        data_lem.append(lem)
    return data_lem

if __name__ == "__main__":

    data_lowercase = []
    # url_all = []
    # data_punct = []
    # data_token = []
    # stopword_data = []
    # ren_htmltags = []
    # sani_stopword = []
    # join_token = []

    lemiti = WordNetLemmatizer()

    # getting dataset
    sentiment_new = pd.read_csv(r'prepd_data.csv', error_bad_lines=False, sep=',')
    sentiment_new.head(10)
    review = sentiment_new['review']
    target = sentiment_new['sentiment_label']

    #pre processing
    lwr_case_positive = lower_case(review)
    review_positive_remhtml = strip_html_tags(lwr_case_positive)
    remove_punct_positive = punctuation(review_positive_remhtml)
    remove_url = url(remove_punct_positive)
    tweet_tokens = token(remove_url)
    remove_stopwords = stopWord(tweet_tokens)
    lemitie_words = lemitization(remove_stopwords)
    join_stopwords = join_tokens(lemitie_words)
    print(join_stopwords)
    # spliting test and train dataset
    X_train, X_test, y_train, y_test = train_test_split(join_stopwords, target, test_size=0.3, random_state=1)
    train_data = X_train
    test_data = X_test
    train_target = y_train
    test_target = y_test

    #
    # # bag of words
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_data)
    X_test_counts = count_vect.transform(test_data)
    #
    # #tfidf
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    #
    # tfidf_transformer_test = TfidfTransformer()
    X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)
    #
    # # scale standard
    sc = StandardScaler(with_mean=False)
    sc.fit(X_train_tfidf)
    X_train_std = sc.transform(X_train_tfidf)
    X_test_std = sc.transform(X_test_tfidf)


    # Instantiate the Support Vector Classifier (SVC)
    svc = LinearSVC(dual=False)

    # Fit the model
    text_svc = svc.fit(X_train_std, train_target)
    # text_svc = svm_clf.fit(train_data, train_target)

    # Make the predictions
    y_predict = svc.predict(X_test_std)
    # y_predict = svm_clf.predict(test_data)

    data = {'Review': test_data, 'labels': y_predict}
    print("working_1")
    df = pd.DataFrame(data)
    positive_data = df.loc[df['labels'] == 1]
    positive_review = positive_data['Review'].tolist()
    positive_words = ' '.join(positive_review)
    negative_data = df.loc[df['labels'] == 0]
    negative_review = negative_data['Review'].tolist()
    negative_words = ' '.join(negative_review)


    classification = classification_report(test_target, y_predict, target_names=['Positive', 'Negative'])
    print(np.mean(y_predict == test_target))

    # Measure the performance
    print("Accuracy score %.3f" % metrics.accuracy_score(test_target, y_predict))
    accuracy = metrics.accuracy_score(test_target, y_predict)

    # Model Precision: what percentage of positive tuples are labeled as such?
    precision = metrics.precision_score(test_target, y_predict)
    print("Precision:", metrics.precision_score(test_target, y_predict))

    # Model Recall: what percentage of positive tuples are labelled as such?
    recall = metrics.recall_score(y_test, y_predict)
    print("Recall:", metrics.recall_score(y_test, y_predict))

    # F1 score
    print("F1 score:", f1_score(y_test, y_predict, average='macro'))

    target_cnt = Counter(target)

    plt.figure(figsize=(16, 8))
    plt.bar(target_cnt.keys(), target_cnt.values())
    plt.title("Dataset labels distribuition")
    plt.savefig('bar_chat')
    plt.show()

    # Confusion matrix

    conf = confusion_matrix(test_target, y_predict)
    print(conf)

    cm = pd.DataFrame(
        conf, index=[i for i in ['0', '1']],
        columns=[i for i in ['0', '1']]
    )

    plt.figure(figsize=(12, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.tight_layout(pad=0)
    plt.savefig('heatmap_svm')
    plt.show()

    plt.figure(figsize=(20, 10))
    WC = WordCloud(width=1600, height=800, background_color="rgba(255, 255, 255, 0)").generate(positive_words)
    plt.imshow(WC)
    plt.axis("off")

    plt.show()
    plt.savefig('wordcloud.png', facecolor='k', bbox_inches='tight')

    WC = WordCloud(width=1600, height=800, background_color="rgba(255, 255, 255, 0)").generate(negative_words)
    plt.imshow(WC)
    plt.axis("off")

    plt.show()
    plt.savefig('wordcloud.png', facecolor='k', bbox_inches='tight')

    # end
    # streamlit

    data_lowercase = []
    url_all = []
    data_punct = []
    data_token = []
    stopword_data = []
    ren_htmltags = []
    sani_stopword = []
    join_token = []


    reviews = pd.read_csv(r'shuffle_dataset.csv')


    review_test_final = reviews['Review']
    remove_html = []
    remove_punct_positive_final = []
    remove_url_final=[]
    tweet_tokens_final = []
    remove_stopwords_final = []

    lower_value = lower_case(review_test_final)
    remove_html = strip_html_tags(lower_value)
    remove_punct_positive_final = punctuation(remove_html)
    remove_url_final = url(remove_punct_positive_final)
    tweet_tokens_final = token(remove_url_final)
    remove_stopwords_final = stopWord(tweet_tokens_final)

    join_token_final = []
    for i in remove_stopwords_final:
        join_words_final = ' '.join(i)
        join_token_final.append(join_words_final)
    X_test_counts = count_vect.transform(join_token_final)

    X_test_std = sc.transform(X_test_counts)

    y_predict_final = svc.predict(X_test_std)
    reviews['lables'] = y_predict_final
    reviews['Status'] = reviews.lables.apply(lambda x: 'Must Watch' if x == 1 else 'Average')
    reviews.to_csv(r'movie_dataset.csv',  encoding='utf-8', header=True)
