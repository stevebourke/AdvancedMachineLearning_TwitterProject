import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re


#Read in the data containing the tweets to dframe
df = pd.read_csv('data\\tweets_all.csv')

#A series of just the text element of the tweets
tweets = df['text']

#Use a reg. ex. to drop all http...
junk_words = re.compile("http.*")
tweets = [junk_words.sub("", line) for line in tweets]

sample_tweet = 'It’s a Steve Coogan laughter kind of weekend! 🙌🏻 The legend was a mystery guest on The Tommy Tiernan show yesterday and will be on Virgin Media One Ireland AM with Simon Delaney in less than 1️⃣ hour. Be sure to tune in! https://t.co/vsMidS2jCj'

#Define functions...
#Tokenize. ie separate out a string into its individual words
def tokenize(txt):
    return word_tokenize(txt)

#Remove basic words such as 'and', 'this', 'the'
def remove_stop_words(txt):
    stop_words = set(stopwords.words('english'))
    return([word for word in txt if word.lower() not in stop_words])

#Remove punctuation marks
def remove_punct(txt):
    puncts = '.,!?£$%^&*"\'()_-+=\|<>/:;@#~[]{}'
    return([char for char in txt if char not in puncts])

#Stem longer forms of words into shorter
def stem_words(txt):
    return [WordNetLemmatizer().lemmatize(word) for word in txt]

#A function which performs all of the above functions at once
def clean_all(txt):
    return stem_words(remove_punct(remove_stop_words(tokenize(txt))))
    
#Separated out so that we can easily see the return from each function
words1 = [tokenize(txt) for txt in tweets]
words2 = [remove_stop_words(word) for word in words1]
words3 = [remove_punct(word) for word in words2]
words4 = [stem_words(word) for word in words3]


#Put the cleaned list of words into a new column in df
df['clean_text'] = words4

#Need to convert list of words in column back to a string for tfidf....
#https://datascience.stackexchange.com/questions/24376/use-of-tfidfvectorizer-on-dataframe
df['clean_text']=[" ".join(text) for text in df['clean_text'].values]

clean_list = df['clean_text']


#Use tfidf to enumerate words in tweets
#tfidf = TfidfVectorizer()
#output = tfidf.fit_transform(df['clean_text'])
#df_out = pd.DataFrame(output.todense(), columns=(tfidf.get_feature_names()))


#clean_list.to_csv('C:\\Users\Admin\\Desktop\\AIT 2021\\AdvancedMachineLearning\\PycharmProjects\\Tweets_clean\\tweets_JGP_lists.csv')