import nltk
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk import FreqDist
import pandas as pd
import nltk
from nltk import classify
from nltk import NaiveBayesClassifier
import base64
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


def lower_case(sanitize):
    data_details_lowercase = sanitize.str.lower()
    return data_details_lowercase


def url(sanitize):
    for word in sanitize:
        data_details_url = re.sub(r"http\S+", "", word)
        url_all.append(data_details_url)
    return url_all


def punctuation(sanitize):
    for i in sanitize:
        data_details_punct = re.sub(r'[^\w\s]', '', i)
        data_punct.append(data_details_punct)
    return data_punct


def token(sanitize):
    for i in sanitize:
        tokenize = nltk.word_tokenize(i)
        data_token.append(tokenize)
    return data_token


def stopWord(sanitize):
    STOPWORDS = set(stopwords.words('english'))
    for k in sanitize:
        for i in k:
            if i not in STOPWORDS:
                print(i)
                sani_stopword.append(i)
    stopword_data.append(sani_stopword)
    return stopword_data


def tokenized(self, fileids=None):
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

        print(cleaned_tokens)

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
        soup = BeautifulSoup(i, "html.parser")
        stripped_text = soup.get_text()
        ren_htmltags.append(stripped_text)
    return ren_htmltags

def punctuation(sanitize):
    for i in sanitize:
        data_details_punct = re.sub(r'[^\w\s]','', i)
        data_punct.append(data_details_punct)
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
    sentiment_new = sentiment_new.head(20)
    sentiment_new['sentiment_words'] = sentiment_new.sentiment.apply(lambda x: 'positive' if x == 1 else 'negative')
    
    positive_tweets_new = sentiment_new[sentiment_new['sentiment_words'].str.contains('positive')]
    negative_tweets_new = sentiment_new[sentiment_new['sentiment_words'].str.contains('negative')]
    
    # positive token
    review_positive = positive_tweets_new['review']
    lwr_case_positive = lower_case(review_positive)
    review_positive_remhtml = strip_html_tags(lwr_case_positive)
    remove_punct_positive = punctuation(review_positive_remhtml)
    review_negative = negative_tweets_new['review']
    lwr_case_negative = lower_case(review_negative)
    review_negative_remhtml = strip_html_tags(lwr_case_negative)
    remove_punct_negative = punctuation(review_negative_remhtml)
    stop_words = stopwords.words('english')

    positive_tweet_tokens = token(remove_punct_positive)
    remove_stopwords_negative = stopWord(positive_tweet_tokens)
    negative_tweet_tokens = token(remove_punct_negative)
    remove_stopwords_negative = stopWord(negative_tweet_tokens)
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    
    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    all_pos_words = get_all_words(positive_cleaned_tokens_list)
    freq_dist_pos = FreqDist(all_pos_words)
    
    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
    
    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]
    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]
    
    dataset = positive_dataset + negative_dataset
    random.shuffle(dataset)

    train_data = dataset[:10]
    test_data = dataset[10:]
    
    classifier = NaiveBayesClassifier.train(train_data)
    print(classify.accuracy(classifier, train_data))

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    data={'Reviews': ['I ordered just once from TerribleCo, they screwed up, never used the app again.','it is beautiful','I am angry','i am not interested in this flim','it worth watching'],
          'Movies': ['wizard of oz','evil dead','alien','Zombie','wall street']}
    custom_tweet = pd.DataFrame(data)
    custom_tweet.to_csv('samplecsv.csv')
    custom_tokens_append = []

    for i in custom_tweet['Reviews']:
        custom_tokens = remove_noise(word_tokenize(i))
        custom_tokens_append.append(custom_tokens)
    original_append = []
    for j in custom_tokens_append:
        print(j)
        original = classifier.classify(dict([token, True] for token in j))
        original_append.append(original)
    custom_tweet['Analysis'] = original_append
    custom_tweet.loc[custom_tweet['Analysis'] == 'Positive', 'Status'] = 'Must Watch'
    custom_tweet.loc[custom_tweet['Analysis'] == 'Negative', 'Status'] = 'Average'
    option = st.selectbox('Select', ('Must Watch', 'Average'))
    st.write('You selected:', option)
    movies = custom_tweet.loc[custom_tweet['Status'] == option, 'Movies']
    list_movie = movies.tolist()
    LOGO_IMAGE = "Logo.jpg"
   
    for i in list_movie:
        st.markdown(i)
        st.markdown(
            """
            <style>
            .container {
                display: flex;
            }
            .logo-text {
                font-weight:700 !important;
                font-size:50px !important;
                color: #f9a01b !important;
                padding-top: 75px !important;
            }
            .logo-img {
                float:right;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="container">
                <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
                <p class="logo-text">Logo Much ?</p>
            </div>
            """,
            unsafe_allow_html=True
        )
   