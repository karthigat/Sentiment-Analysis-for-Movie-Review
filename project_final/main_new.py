import nltk
# from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk import FreqDist
# nltk.download('twitter_samples')
# nltk.download('averaged_perceptron_tagger')
import pandas as pd
import nltk
from nltk import classify
from nltk import NaiveBayesClassifier
import base64
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
# from wordcloud import WordCloud
# import dask.dataframe as dd
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
    print(tweet_tokens)

    for token, tag in pos_tag(tweet_tokens):
        print('working')
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
    # print(dataset)
    # print(dataset)
    random.shuffle(dataset)

    train_data = dataset[:10]
    test_data = dataset[10:]
    # print(train_data)
    # print(test_data)

    classifier = NaiveBayesClassifier.train(train_data)
    print(classify.accuracy(classifier, train_data))

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    # print(classifier.show_most_informative_features(10))

    # custom_tweet = ["I ordered just once from TerribleCo, they screwed up, never used the app again.","it is beautiful","I am angry","i am not interested in this flim"]
    data={'Reviews': ['I ordered just once from TerribleCo, they screwed up, never used the app again.','it is beautiful','I am angry','i am not interested in this flim','it worth watching'],
          'Movies': ['wizard of oz','evil dead','alien','Zombie','wall street']}
    # original = []
    # # append_removenoise = []
    # custom_tokens = []
    custom_tweet = pd.DataFrame(data)
    #
    custom_tweet.to_csv('samplecsv.csv')
    print(custom_tweet['Reviews'])
    # print(custom_tweet)

    # custom_tweet = pd.read_csv(r'movie_review.xlsx', error_bad_lines=False)
    # custom_tweet.to_csv(r'movie_review_new.csv', index=None)

    # custom_tweet = custom_tweet['Reviews']

    custom_tokens_append = []

    # tokeni = token(custom_tweet)
    # print(tokeni)
    # for i in tokeni:
    #     custom_tokens.append(remove_noise(tokeni))
    #     # custom_tokens = remove_noise(tokeni)
    # print(custom_tokens)
    # tokenizing = token(custom_tweet)
    for i in custom_tweet['Reviews']:
        print(i)
        custom_tokens = remove_noise(word_tokenize(i))
        # custom_tokens = remove_noise(word_tokenize(custom_tweet))
        custom_tokens_append.append(custom_tokens)
    # custom_tokens = remove_noise(tokenizing)
    print(custom_tokens_append)
    # for j in custom_tokens:
    #     print("workingj")
    #     print(j)
    #     print(custom_tweet, classifier.classify(dict([token_1, True] for token_1 in j)))
    original_append = []
    for j in custom_tokens_append:
        print(j)
        original = classifier.classify(dict([token, True] for token in j))
        print(original)
        # if original == "Positive":
        #     print('true')
        #     custom_tweet['Status'] = "Must Watch"
        #     print(custom_tweet['Status'])
        # else:
        #     custom_tweet['Status'] = "Average"

        original_append.append(original)
    custom_tweet['Analysis'] = original_append
    print(custom_tweet)
    # sentiment_new[sentiment_new['sentiment_words'].str.contains('positive')]
    # custom_tweet_check = custom_tweet[custom_tweet['Analysis'].str.contains('Positive')]
    custom_tweet.loc[custom_tweet['Analysis'] == 'Positive', 'Status'] = 'Must Watch'
    custom_tweet.loc[custom_tweet['Analysis'] == 'Negative', 'Status'] = 'Average'
    print(custom_tweet)
    option = st.selectbox('Select', ('Must Watch', 'Average'))
    print(option)
    st.write('You selected:', option)
    movies = custom_tweet.loc[custom_tweet['Status'] == option, 'Movies']
    # print((movies).array)
    # movies_array = movies.array
    # html_list = "<li></li>"
    list_movie = movies.tolist()
    # html = "<h6>select name</h6>"
    # st.markdown(html,  unsafe_allow_html=True)
    LOGO_IMAGE = "Logo.jpg"
    #
    # st.markdown(
    #     """
    #     <style>
    #     .container {
    #         display: flex;
    #     }
    #     .logo-text {
    #         font-weight:700 !important;
    #         font-size:50px !important;
    #         color: #f9a01b !important;
    #         padding-top: 75px !important;
    #     }
    #     .logo-img {
    #         float:right;
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )
    #
    # st.markdown(
    #     f"""
    #     <div class="container">
    #         <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
    #         <p class="logo-text">Logo Much ?</p>
    #     </div>
    #     """,
    #     unsafe_allow_html=True
    # )
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
    # st.write(movies_array)
    # if custom_tweet_check:
    #     custom_tweet['Status'] = "Must Watch"
    # else:
    #     custom_tweet['Status'] = "Average"
    # print(original_append)
    # original = custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens))
    # original = classifier.classify(dict([token, True] for token in custom_tokens))
    #     original.append(classifier.classify(dict([token_1, True] for token_1 in j)))
    # print(original)
    # custom_result.append(original)
    # print(custom_result)
    # custom_tweet['analysis'] = original.to_string()
    # if 'positive' in custom_tweet['analysis']:
        # if original == 'Positive':
        #     custom_tweet['status'] = 'Must Watch'
        # else:
        #     custom_tweet['status'] = 'Average'

    # st.write(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))