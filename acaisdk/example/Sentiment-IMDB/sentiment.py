import sys
import pandas as pd
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)


def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)
    return text


def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


inputs = sys.argv[1]
outputs = sys.argv[2]

imdb_data = pd.read_csv(inputs + 'IMDB Dataset.csv')

train_reviews = imdb_data.review[:40000]
train_sentiments = imdb_data.sentiment[:40000]

test_reviews = imdb_data.review[40000:]
test_sentiments = imdb_data.sentiment[40000:]

nltk.download('stopwords')
global stopword_list
stopword_list = nltk.corpus.stopwords.words('english')

# Text normalization
tokenizer = ToktokTokenizer()
imdb_data['review'] = imdb_data['review'].apply(denoise_text)
imdb_data['review'] = imdb_data['review'].apply(remove_special_characters)
imdb_data['review'] = imdb_data['review'].apply(simple_stemmer)
# set stopwords to english
imdb_data['review'] = imdb_data['review'].apply(remove_stopwords)

norm_train_reviews = imdb_data.review[:40000]
norm_test_reviews = imdb_data.review[40000:]

# Count vectorizer for bag of words
cv = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1, 3))
cv_train_reviews = cv.fit_transform(norm_train_reviews)

cv_test_reviews = cv.transform(norm_test_reviews)

# Tfidf vectorizer
tv = TfidfVectorizer(min_df=0, max_df=1, use_idf=True, ngram_range=(1, 3))
tv_train_reviews = tv.fit_transform(norm_train_reviews)
tv_test_reviews = tv.transform(norm_test_reviews)

# Labeling the sentient data
lb = LabelBinarizer()
sentiment_data = lb.fit_transform(imdb_data['sentiment'])

train_sentiments = sentiment_data[:40000]
test_sentiments = sentiment_data[40000:]

# Modelling the dataset
# Logistic regression model for both bag of words and tfidf features
lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
lr_bow = lr.fit(cv_train_reviews, train_sentiments)
lr_tfidf = lr.fit(tv_train_reviews, train_sentiments)

# Logistic regression model performance on test dataset
lr_bow_predict = lr.predict(cv_test_reviews)
lr_tfidf_predict = lr.predict(tv_test_reviews)

lr_bow_score = accuracy_score(test_sentiments, lr_bow_predict)
lr_tfidf_score = accuracy_score(test_sentiments, lr_tfidf_predict)

# Stochastic gradient descent or Linear support vector machines for bag of words and tfidf features
svm = SGDClassifier(loss='hinge', max_iter=500, random_state=42)
svm_bow = svm.fit(cv_train_reviews, train_sentiments)
svm_tfidf = svm.fit(tv_train_reviews, train_sentiments)
svm_bow_predict = svm.predict(cv_test_reviews)
svm_tfidf_predict = svm.predict(tv_test_reviews)

svm_bow_score = accuracy_score(test_sentiments, svm_bow_predict)
svm_tfidf_score = accuracy_score(test_sentiments, svm_tfidf_predict)

# Multinomial Naive Bayes for bag of words and tfidf features
mnb = MultinomialNB()
mnb_bow = mnb.fit(cv_train_reviews, train_sentiments)
mnb_tfidf = mnb.fit(tv_train_reviews, train_sentiments)
mnb_bow_predict = mnb.predict(cv_test_reviews)
mnb_tfidf_predict = mnb.predict(tv_test_reviews)
mnb_bow_score = accuracy_score(test_sentiments, mnb_bow_predict)
mnb_tfidf_score = accuracy_score(test_sentiments, mnb_tfidf_predict)

with open(outputs + '/scores.txt', 'w') as f:
    f.write("lr_bow_score :" + str(lr_bow_score) + "\n")
    f.write("lr_tfidf_score :" + str(lr_tfidf_score) + "\n")
    f.write("svm_bow_score :" + str(svm_bow_score) + "\n")
    f.write("svm_tfidf_score :" + str(svm_tfidf_score) + "\n")
    f.write("mnb_bow_score :" + str(mnb_bow_score) + "\n")
    f.write("mnb_tfidf_score :" + str(mnb_tfidf_score))
