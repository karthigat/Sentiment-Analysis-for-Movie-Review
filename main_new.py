import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

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


if __name__ == "__main__":

    data_lowercase = []
    url_all = []
    data_punct = []
    data_token = []
    stopword_data = []

    sentiment_new = pd.read_csv(r'prepd_data.csv', sep=",")
    # print(sentiment_new)
    positive_tweets_new = sentiment_new[sentiment_new['sentiment'].str.contains('positive')]
    # print(positive_tweets_new)
    negative_tweets_new = sentiment_new[sentiment_new['sentiment'].str.contains('negative')]

    # positive token
    review_positive = positive_tweets_new['review']
    lwr_case_positive = lower_case(review_positive)
    tokens_positive = token(lwr_case_positive)
    print(tokens_positive)

    rm_url_positive = url(lwr_case_positive)
    punct_positive = punctuation(rm_url_positive)
    tokens_positive = token(punct_positive)  # array of words
    stop_word_positive = stopWord(tokens_positive)

    # negative tokens
    review_negative = negative_tweets_new['review']
    lwr_case = lower_case(review_negative)
    tokens_negative = token(lwr_case)
    rm_url_negative = url(lwr_case)
    punct_negative = punctuation(rm_url_negative)
    tokens_negative = token(punct_negative)  # array of words
    stop_word_negative = stopWord(tokens_negative)

    # print(stop_word)
    print("working_1")
    # print(negative_tweets_new)
    # print("")
    # positive_tweets = twitter_samples.strings('positive_tweets.json')
    # print(positive_tweets)
    # negative_tweets = twitter_samples.strings('negative_tweets.json')
    # print(negative_tweets)
    # text = twitter_samples.strings('tweets.20150430-223406.json')
    # positive_tweet_token = positive_tweets_new.tokenized(positive_tweets_new)
    # print(positive_tweet_token)
    tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]
    # print(tweet_tokens)
    stop_words = stopwords.words('english')
    # print(stop_words)

    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    # print(positive_tweet_tokens)
    # positive_tweet_tokens = token(positive_tweets_new)
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')
    # print(negative_tweet_tokens)
    # negative_tweet_tokens = token(negative_tweets_new)
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    print("working_2")
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

    positive_tokens_for_model = get_tweets_for_model(tokens_positive)
    # positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    # print(positive_tokens_for_model)
    negative_tokens_for_model = get_tweets_for_model(tokens_negative)
    # negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
    # print(negative_tokens_for_model)

    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]
    # print(positive_dataset)
    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]
    # print(negative_dataset)

    dataset = positive_dataset + negative_dataset
    # print(dataset)
    # print(dataset)
    random.shuffle(dataset)

    train_data = dataset[:25000]
    test_data = dataset[25000:]

    classifier = NaiveBayesClassifier.train(train_data)
    print(classify.accuracy(classifier, train_data))

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    # print(classifier.show_most_informative_features(10))

    custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."

    custom_tokens = remove_noise(word_tokenize(custom_tweet))

    # print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))