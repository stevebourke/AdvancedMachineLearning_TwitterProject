from flask import Flask, render_template, request

# Import twitter authentication key and tokens
from twitter_auth import API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET

import tweepy as tp
import pandas as pd

auth = tp.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

# Set up the api passing in the authenticator
api = tp.API(auth)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def process_query():
    if request.method == 'POST':
        query = request.form['query']
        tweets = api.search_tweets(q=query, result_type='recent', count=100, tweet_mode='extended')

        output = []
        for tweet in tweets:
            text = tweet.full_text
            favourite_count = tweet.favorite_count
            retweet_count = tweet.retweet_count
            created_at = tweet.created_at
            location = tweet.user.location

            line = {'favourite_count': favourite_count, 'retweet_count': retweet_count,
                'created_at': created_at, 'location': location, 'text': text}
            output.append(line)

        df = pd.DataFrame(output)
        df.to_csv('tweets_11.csv', encoding='utf-8')

        return render_template('results.html', tweets=tweets)
    else:
        return render_template('index.html')

    # Method taken from  https://towardsdatascience.com/how-to-build-a-dataset-from-twitter-using-python-tweepy

    # Create a list for our tweets



if __name__ == '__main__':
    app.run()
