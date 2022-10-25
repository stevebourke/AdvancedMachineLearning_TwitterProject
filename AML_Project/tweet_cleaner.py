import pandas as pd
import regex as re


df = pd.read_csv('data\\tweets_all.csv')

#Remove duplicate tweets
#df = df.drop_duplicates(subset='text', keep="first")

#Drop unsuitable tweets
#df.drop(df.columns[[2,3]], axis=1, inplace=True)

#Use these as our sentiment labels
p = "Positive"
n = "Negative"
o = "Neutral"

#Create new sentiment column and fill it with a value for each tweet
#df['sentiment'] = [n,p,p,p,o,n,o,p,p,n,n,n,n,p,n,p,p,p,o,o,n,p,p,n,p,n,p,p,n,n,n,n,n,n,p,p,n,n,n,n,p,n,n,p,p,n,p,p]

#Return the updated dataframe to the file
#df.to_csv('C:\\Users\Admin\\Desktop\\AIT 2021\\AdvancedMachineLearning\\PycharmProjects\\Tweets_clean\\tweets_Burren_clean.csv')

banned_list = ['(?i)Varadkar', '(?i)Pancake', '(?i)Qanon', '(?i)Burren', '(?i)Daniel Kinahan', '(?i)Eoghan McDermott',
               '(?i)Shamrock Rovers', '(?i)Eastenders', "(?i)Tommy Tiernan", "(?i)Sinn Fein", "(?i)Kinahan", "(?i)Dan", "(?i)Daniel"
               '2', '“', '’', '(?i)rt', '(?i)leo', '(?i)shamrockrovers','``', "''", "'s", "n't", 'amp', 'show', '(?i)sinn', '(?i)féin', 'peakybannsiders']
banned = re.compile('|'.join(banned_list))

tweets = df['clean_text']

tweets = [banned.sub("", line) for line in tweets]

df['clean_text'] = tweets

#df.to_csv('C:\\Users\Admin\\Desktop\\DataVisProject\\data\\tweets_all.csv')