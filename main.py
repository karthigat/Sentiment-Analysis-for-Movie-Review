import csv  # we need import file
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import dask.dataframe as dd



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
        data_details_punct = re.sub(r'[^\w\s]','', i)
        data_punct.append(data_details_punct)
        # data_details_punct = i.str.replace('[^\w\s]','')
    return data_punct

def token(sanitize):
    for i in sanitize:
        tokenize = nltk.word_tokenize(i)
        data_token.append(tokenize)
    return data_token

def stopWord(sanitize):
    STOPWORDS = set(stopwords.words('english'))
    print("tokenize")
    # print(tok)
    for k in sanitize:
        # print(k)
        # for i in k:
        #     print(i)
        #     if i not in STOPWORDS:
        #         print(i)
        s_words = [w for w in k if w not in STOPWORDS]
        stopword_data.append(s_words)
        # print(s_words)
    return stopword_data

def lemitization(sanitize):
    for j in sanitize:
        lem = [lemiti.lemmatize(i) for i in j]
        data_lem.append(lem)
    return data_lem

def englishwords(sanitize):
    eng_words = set(nltk.corpus.words.words())
    # k = 'Io andiamo to the beach with my amico'
    for i in sanitize:
        engwords = [w for w in nltk.wordpunct_tokenize(i) if w in eng_words or not w.isalpha()]
        engwords = str(engwords)
        english_words.append(engwords)
    # print(english_words)
    return english_words


if __name__ == "__main__":
    data_lowercase = []
    url_all = []
    data_punct = []
    data_token = []
    stopword_data = []
    data_lem = []
    data_sent = []
    single = []
    bigrams = []
    english_words = []
    comment = []


    sentimet = pd.read_csv(r'prepd_data.csv', sep=",")
    # sentimet.encode('utf-8').strip()
    review = sentimet['review']
    lwr_case = lower_case(review)
    rm_url = url(lwr_case)
    punct = punctuation(rm_url)
    tokens = token(punct) #array of words
    stop_word = stopWord(tokens) #
    print(url)
