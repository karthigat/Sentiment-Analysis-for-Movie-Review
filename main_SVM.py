import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

# Sklearn modules & classes
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import metrics

nltk.download('twitter_samples')
nltk.download('averaged_perceptron_tagger')
import pandas as pd
import re
import nltk
from nltk import classify
from nltk import NaiveBayesClassifier

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# import dask.dataframe as dd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import re, string, random
from bs4 import BeautifulSoup


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
    STOPWORDS = set(stopwords.words('english'))
    word = "roses are red"
    print(STOPWORDS)
    print("tokenize")
    # print(sanitize)
    # lwr_case = sanitize.str.lower()
    for k in sanitize:
        # print(k)
        for i in k:
            if i not in STOPWORDS:
                print(i)
                sani_stopword.append(i)
    stopword_data.append(sani_stopword)
            # print(i)

    # for k in word:
    #     print(k)
    # if sanitize not in STOPWORDS:
    #     print(sanitize)
        # print(k)


    # print(tok)
    # for k in sanitize:
    #     print(k)
    #     for i in k:
    #         print(i)
    #         if i not in STOPWORDS:
    #             print(i)

    # s_words = [w for w in sanitize if w not in STOPWORDS]
    # print(s_words)
                # stopword_data.append(s_words)
                # stopword_data.append(i)

    # print(s_words)
    return stopword_data


def tokenized(self, fileids=None):
    """
    :return: the given file(s) as a list of the text content of Tweets as
    as a list of words, screenanames, hashtags, URLs and punctuation symbols.

    :rtype: list(list(str))
    """
    tweets = self.strings(fileids)
    tokenizer = self._word_tokenizer
    return [tokenizer.tokenize(t) for t in tweets]


def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())

    return cleaned_tokens


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

def strip_html_tags(text):
    for i in text:
        # print(i)
        soup = BeautifulSoup(i, "html.parser")
        # print(soup)
        stripped_text = soup.get_text()
        ren_htmltags.append(stripped_text)
        # print(ren_htmltags)
    return ren_htmltags

def punctuation(sanitize):
    for i in sanitize:
        data_details_punct = re.sub(r'[^\w\s]','', i)
        data_punct.append(data_details_punct)
        # print(data_punct)
        # data_details_punct = i.str.replace('[^\w\s]','')
    return data_punct


if __name__ == "__main__":

    data_lowercase = []
    url_all = []
    data_punct = []
    data_token = []
    stopword_data = []
    ren_htmltags = []
    sani_stopword = []

    sentiment_new = pd.read_csv(r'IMDB_dataset.csv', error_bad_lines=False, sep=',')
    # sentiment_new.columns = ['sno','review','sentiment']
    sentiment_new = sentiment_new.head(20)
    sentiment_new['sentiment_words'] = sentiment_new.sentiment.apply(lambda x: 'positive' if x == 1 else 'negative')
    # print(sentiment_new)

    # positive_tweets_new = sentiment_new[sentiment_new['sentiment'].str.contains('positive')]
    positive_tweets_new = sentiment_new[sentiment_new['sentiment_words'].str.contains('positive')]
    # print(positive_tweets_new)
    # positive_tweets_new = sentiment_new[sentiment_new['sentiment'].astype(int).contains(1)]
    # print(positive_tweets_new)
    # negative_tweets_new = sentiment_new[sentiment_new['sentiment'].str.contains('negative')]
    negative_tweets_new = sentiment_new[sentiment_new['sentiment_words'].str.contains('negative')]
    # print(negative_tweets_new)

    # positive token
    review_positive = positive_tweets_new['review']
    lwr_case_positive = lower_case(review_positive)
    review_positive_remhtml = strip_html_tags(lwr_case_positive)
    remove_punct_positive = punctuation(review_positive_remhtml)
    # print(review_positive_remhtml)
    # lwr_case_positive = lower_case(review_positive)
    # # tokens_positive = token(lwr_case_positive)
    # # print(tokens_positive)
    #
    # rm_url_positive = url(lwr_case_positive)
    # punct_positive = punctuation(rm_url_positive)
    # tokens_positive = token(punct_positive)  # array of words
    # stop_word_positive = stopWord(tokens_positive)

    # negative tokens
    review_negative = negative_tweets_new['review']
    # remove_stopwords_negative = stopWord(review_negative)
    lwr_case_negative = lower_case(review_negative)
    review_negative_remhtml = strip_html_tags(lwr_case_negative)
    # print(review_negative_remhtml)
    remove_punct_negative = punctuation(review_negative_remhtml)
    # lwr_case = lower_case(review_negative)
    # tokens_negative = token(lwr_case)
    # rm_url_negative = url(lwr_case)
    # punct_negative = punctuation(rm_url_negative)
    # tokens_negative = token(punct_negative)  # array of words
    # stop_word_negative = stopWord(tokens_negative)

    # positive_tweets = twitter_samples.strings('positive_tweets.json')
    # negative_tweets = twitter_samples.strings('negative_tweets.json')
    # text = twitter_samples.strings('tweets.20150430-223406.json')
    # positive_tweet_token = positive_tweets_new.tokenized(positive_tweets_new)
    # tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]
    stop_words = stopwords.words('english')

    # positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    positive_tweet_tokens = token(remove_punct_positive)
    # print(positive_tweet_tokens)
    remove_stopwords_negative = stopWord(positive_tweet_tokens)
    # negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
    negative_tweet_tokens = token(remove_punct_negative)
    # print(negative_tweet_tokens)
    remove_stopwords_negative = stopWord(negative_tweet_tokens)
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    # print(positive_cleaned_tokens_list)

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    # print(positive_cleaned_tokens_list)
    all_pos_words = get_all_words(positive_cleaned_tokens_list)
    # print(all_pos_words.encode)
    freq_dist_pos = FreqDist(all_pos_words)
    # print(freq_dist_pos.most_common(10))

    # positive_tokens_for_model = get_tweets_for_model(tokens_positive)
    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    # print(positive_tokens_for_model)
    # negative_tokens_for_model = get_tweets_for_model(tokens_negative)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
    # print(negative_tokens_for_model)

    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]
    # print(positive_dataset)
    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]
    # print(negative_dataset)

    dataset = positive_dataset + negative_dataset
    print(dataset)
    # print(dataset)
    random.shuffle(dataset)

    #train_data = dataset[:10]
   # test_data = dataset[10:]
    # print(train_data)
    # print(test_data)

    #Y = pd.read_csv(r'IMDB_dataset.csv', index_col=1)

    # Create training and test split
    X_train, X_test, y_train, y_test = train_test_split(dataset['review'], dataset['sentiment'], test_size=0.3, random_state=1, stratify=Y)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # Instantiate the Support Vector Classifier (SVC)
    from sklearn.svm import LinearSVC
    svc = LinearSVC(dual=False)
  #  svc = SVC(C=1.0, random_state=1, kernel='linear')

    # Fit the model
    svc.fit(X_train_std, y_train)

    # Make the predictions
    y_predict = svc.predict(X_test_std)

    # Measure the performance
    print("Accuracy score %.3f" % metrics.accuracy_score(y_test, y_predict))

    #classifier = NaiveBayesClassifier.train(train_data)
   # print(classify.accuracy(classifier, train_data))

  #  print("Accuracy is:", classify.accuracy(classifier, test_data))

       # print(classifier.show_most_informative_features(10))

    # custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."
  #  custom_tweet = "The moview was genuinely awesome"

   # custom_tokens = remove_noise(word_tokenize(custom_tweet))

   # print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))