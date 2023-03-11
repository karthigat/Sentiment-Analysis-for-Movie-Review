from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.svm import LinearSVC
import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from sklearn.metrics import f1_score

# converting to lower case
def lower_case(sanitize):
    data_lowercase = []
    data_details_lowercase = sanitize.str.lower()
    return data_details_lowercase

# remove url
def url(sanitize):
    url_all = []
    for word in sanitize:
        data_details_url = re.sub(r"http\S+", "", word)
        url_all.append(data_details_url)
    return url_all

#remove punctuation
def punctuation(sanitize):
    data_punct = []
    for i in sanitize:
        data_details_punct = re.sub(r'[^\w\s]', '', i)
        data_punct.append(data_details_punct)
    return data_punct

#tokenize
def token(sanitize):
    data_token = []
    for i in sanitize:
        tokenize = nltk.word_tokenize(i)
        data_token.append(tokenize)
    return data_token

#remove stopwords
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

#remove html tags
def strip_html_tags(text):
    ren_htmltags = []
    for i in text:
        soup = BeautifulSoup(i, "html.parser")
        stripped_text = soup.get_text()
        ren_htmltags.append(stripped_text)
    return ren_htmltags

#lemitize the words
def lemitization(sanitize):
    data_lem = []
    for j in sanitize:
        lem = [lemiti.lemmatize(i) for i in j]
        data_lem.append(lem)
    return data_lem

#join tokens
def join_tokens(remove_stopwords):
    join_token = []
    for i in remove_stopwords:
        join_words = ' '.join(i)
        join_token.append(join_words)
    return join_token


def join_positive(positive):
    positive_join = []
    for i in positive:
        join_data = ' '.join(i)
        positive_join.append(join_data)
    return positive_join

if __name__ == "__main__":


    data_lowercase = []

    lemiti = WordNetLemmatizer()

    # getting dataset
    sentiment_new = pd.read_csv(r'IMDB_Dataset.csv', error_bad_lines=False, sep=',')
    sentiment_new.head(10)
    review = sentiment_new['review']
    target = sentiment_new['sentiment_label']

    print(".........Processing........")

    #pre processing
    lwr_case_positive = lower_case(review)
    review_positive_remhtml = strip_html_tags(lwr_case_positive)
    remove_punct_positive = punctuation(review_positive_remhtml)
    remove_url = url(remove_punct_positive)
    tweet_tokens = token(remove_url)
    remove_stopwords = stopWord(tweet_tokens)
    lemitie_words = lemitization(remove_stopwords)
    join_stopwords = join_tokens(lemitie_words)

    # spliting test and train dataset
    X_train, X_test, y_train, y_test = train_test_split(join_stopwords, target, test_size=0.3, random_state=1)
    train_data = X_train
    test_data = X_test
    train_target = y_train
    test_target = y_test


    # bag of words
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(train_data)
    X_test_counts = count_vect.transform(test_data)

    # #tfidf
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    # tfidf_transformer_test = TfidfTransformer()
    X_test_tfidf = tfidf_transformer.fit_transform(X_test_counts)

    #scale standard
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

    # streamlit

    data_lowercase = []
    url_all = []
    data_punct = []
    data_token = []
    stopword_data = []
    ren_htmltags = []
    sani_stopword = []
    join_token = []


    st_reviews = pd.read_csv(r'Movie_images\Streamlit_movieDataset.csv')


    st_review = st_reviews['Review']


    st_lower_value = lower_case(st_review)
    st_remove_html = strip_html_tags(st_lower_value)
    st_remove_punct = punctuation(st_remove_html)
    st_remove_url = url(st_remove_punct)
    st_tweet_tokens = token(st_remove_url)
    st_remove_stopwords = stopWord(st_tweet_tokens)
    st_lemitize = lemitization(st_remove_stopwords)
    st_join_token = join_tokens(st_lemitize)

    # BOW
    X_test_counts = count_vect.transform(st_join_token)
    # tfidf
    X_test_tfidf_streamlit = tfidf_transformer.fit_transform(X_test_counts)
    # sc
    X_test_std = sc.transform(X_test_tfidf_streamlit)
    # svc
    y_predict_final = svc.predict(X_test_std)

    st_reviews['lables'] = y_predict_final
    st_reviews['Status'] = st_reviews.lables.apply(lambda x: 'Must Watch' if x == 1 else 'Average')
    st_reviews.to_csv(r'Movie_images\movie_dataset_svm.csv',  encoding='utf-8', header=True)

    # plot

    target_cnt = Counter(target)

    # Confusion matrix

    conf = confusion_matrix(test_target, y_predict)

    cm = pd.DataFrame(
        conf, index=[i for i in ['0', '1']],
        columns=[i for i in ['0', '1']]
    )

    plt.figure(figsize=(12, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.tight_layout(pad=0)
    plt.savefig('heatmap_svm.png')


    plt.figure(figsize=(20, 10))
    WC = WordCloud(width=1600, height=800, background_color="rgba(255, 255, 255, 0)").generate(positive_words)
    plt.imshow(WC)
    plt.axis("off")


    plt.savefig('wordcloud_positive_svm.png', facecolor='k', bbox_inches='tight')

    WC = WordCloud(width=1600, height=800, background_color="rgba(255, 255, 255, 0)").generate(negative_words)
    plt.imshow(WC)
    plt.axis("off")

  
    plt.savefig('wordcloud_negaitive_svm.png', facecolor='k', bbox_inches='tight')

    print("........completed........")
