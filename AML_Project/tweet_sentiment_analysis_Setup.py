
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

import pandas as pd


#Needed for tfidf command below
def dummy(doc):
    return doc


#Read in three dataframes...Politics batch...Accuracy = 61.54% with Multinomial NB
df1 = pd.read_csv('C:\\Users\Admin\\Desktop\\AIT 2021\\AdvancedMachineLearning\\PycharmProjects\\Tweets_clean\\tweets_Varadkar_clean.csv')
df2 = pd.read_csv('C:\\Users\Admin\\Desktop\\AIT 2021\\AdvancedMachineLearning\\PycharmProjects\\Tweets_clean\\tweets_SF_clean.csv')
df3 = pd.read_csv('C:\\Users\Admin\\Desktop\\AIT 2021\\AdvancedMachineLearning\\PycharmProjects\\Tweets_clean\\tweets_QAnon_clean.csv')

#...TV batch...Accuracy = 59.565% with Multinomial NB
df4 = pd.read_csv('C:\\Users\Admin\\Desktop\\AIT 2021\\AdvancedMachineLearning\\PycharmProjects\\Tweets_clean\\tweets_Eastenders_clean.csv')
df5 = pd.read_csv('C:\\Users\Admin\\Desktop\\AIT 2021\\AdvancedMachineLearning\\PycharmProjects\\Tweets_clean\\tweets_TTiernan_clean.csv')
df6 = pd.read_csv('C:\\Users\Admin\\Desktop\\AIT 2021\\AdvancedMachineLearning\\PycharmProjects\\Tweets_clean\\tweets_EMcDermott_clean.csv')

#...Others batch...Accuracy = 51.35% with Multinomial NB
df7 = pd.read_csv('C:\\Users\Admin\\Desktop\\AIT 2021\\AdvancedMachineLearning\\PycharmProjects\\Tweets_clean\\tweets_Pancakes_clean.csv')
df8 = pd.read_csv('C:\\Users\Admin\\Desktop\\AIT 2021\\AdvancedMachineLearning\\PycharmProjects\\Tweets_clean\\tweets_Burren_clean.csv')
df9 = pd.read_csv('C:\\Users\Admin\\Desktop\\AIT 2021\\AdvancedMachineLearning\\PycharmProjects\\Tweets_clean\\tweets_DKinahan_clean.csv')
df10 = pd.read_csv('C:\\Users\Admin\\Desktop\\AIT 2021\\AdvancedMachineLearning\\PycharmProjects\\Tweets_clean\\tweets_SRovers_clean.csv')

#Merge the batches into one frame each
df_politics = pd.concat([df1,df2,df3])
df_TV = pd.concat([df4,df5,df6])
df_others = pd.concat([df7,df8,df9,df10])

#Create a frame for all topics together
df_all = pd.concat([df_politics, df_TV, df_others])

#X is the cleaned tweet and y is the sentiment
X,y = df2['clean_text'], df2['sentiment']

#Split into train and test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=32, stratify=y)
#If we want to use one model on a different set of tweets...
# x_train = df_politics['clean_text']
# x_test = df_TV['clean_text']
# y_train = df_politics['sentiment']
# y_test = df_TV['sentiment']

tfidf = TfidfVectorizer(analyzer='word', tokenizer=dummy, preprocessor=dummy, token_pattern=None, ngram_range=(1,1))

x_train1 = tfidf.fit_transform(x_train)
x_test1 = tfidf.transform(x_test)


nb_clf =  MultinomialNB(alpha=0.005)

nb_clf.fit(x_train1, y_train)

print(nb_clf.score(x_test1, y_test))

y_pred = nb_clf.predict(x_test1)

print(classification_report(y_test, y_pred))

print(pd.crosstab(y_test, y_pred)) 

# Qanon_list = [55, 40, 39, 16, 6, 10, 51, 33, 41, 29, 48, 1]
# Qanon_list.sort()
# df_Qanon = df3.iloc[Qanon_list,[8,9,10,11]]
# df_Qanon['tfidf'] = y_pred
#df_SF.to_csv('C:\\Users\Admin\\Desktop\\AIT 2021\\AdvancedMachineLearning\\PycharmProjects\\Tweets_clean\\tweets_tfidf_SF.csv')

