from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.metrics import classification_report

#Get the tweets from csv into dataframe
df = pd.read_csv('data\\tweets_all.csv')

#Set up a list for each setiment component
pos_list = []
neg_list = []
neu_list = []
comp_list = []
binary_list = []

#Use Vader to get a figure for sentiment analysis of each batch of tweets
#Break up the components of Vader into 4 lists
analyzer = SentimentIntensityAnalyzer()
for tweet in df['clean_text']:
    vader_sent = analyzer.polarity_scores(tweet)
    pos_list.append(vader_sent.get('pos'))
    neg_list.append(vader_sent.get('neg'))
    neu_list.append(vader_sent.get('neu'))
    comp_list.append(vader_sent.get('compound'))

#Split comp list into positive and negative groupings
for comp in comp_list:
    if comp < 0:
        binary_list.append("Negative")
    elif comp > 0:
        binary_list.append("Positive")
    else:
        binary_list.append("Neutral")

#Add a new column to df for each list
df['vader_pos'] = pos_list
df['vader_neg'] = neg_list
df['vader_neu'] = neu_list
df['vader_comp'] = comp_list
df['vader_binary'] = binary_list

print(classification_report(df['sentiment'], df['vader_binary']))

print(pd.crosstab(df['sentiment'], df['vader_binary']))

#Return the updated dataframe to the file
#df.to_csv('C:\\Users\Admin\\Desktop\\AIT 2021\\AdvancedMachineLearning\\PycharmProjects\\Tweets_clean\\tweets_all.csv')