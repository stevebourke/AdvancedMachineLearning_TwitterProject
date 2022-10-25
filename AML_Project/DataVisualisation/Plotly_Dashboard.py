#### Seems to load better in Firefox than in Chrome ###

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dash import Dash, dcc, html, Input, Output
from collections import Counter

# Import the csv file
df = pd.read_csv('data\\tweets_all.csv')

# Create a set for the tweet topics
topics = list(set(df['topic']))


# Function for getting most common words from each tweet subject
def get_common_words(subject):
    
    df_subject = df[df['topic'] == subject]
    
    # Create a dictionary for the words contained in cleaned tweets
    tweet_dict = {}
    
    # Convert text column into a list
    tweet_list = df_subject['clean_text'].tolist()
    
    # Add words to dictionary and keep count of occurrence of each
    for tweet in tweet_list:
        
        for word in tweet.split():
            
            if word.lower() not in tweet_dict:           
                tweet_dict[word.lower()] = 1
                
            else:           
                tweet_dict[word.lower()] += 1
                
    
    # Get twenty most common words from dictionary
    common_words = dict(Counter(tweet_dict).most_common(20))
    
    # Sort alphabetically rather than by value - looks better in heatmap
    cw_sorted = dict(sorted(common_words.items()))
    
    # Split into the word and a list of lists of values - needed for heatmap
    words = list(cw_sorted.keys())
    nums = list(cw_sorted.values())
    
    # Taken from https://www.delftstack.com/howto/python/python-split-list-into-multiple-lists/
    words_lists = [words[x:x+5] for x in range(0, len(words), 5)]
    nums_lists = [nums[x:x+5] for x in range(0, len(nums), 5)]
    
    return words_lists, nums_lists




# Information drawn from https://plotly.com/python/getting-started/
# and https://appsilon.com/dash-vs-shiny/
app = Dash(__name__)

# Create the layout for the pie chart - note use of html style code
app.layout = html.Div([ # Div to hold all graphs
    html.Div(className='div_container', id='cont1',
             children=[
                     
    html.Aside(className='aside_panel',
             children=[
    html.H3('Popularity of Tweet'), #Heading
    
    html.P('Choose Topic / Location:'),
    dcc.RadioItems(id='radio1', #Radio buttons
                  options = [{'label':'Topics', 'value':'topic'},
                             {'label':'Location', 'value':'location'}],
                  value='topic'
                  ),
    
    html.P('Retweeted / Favourited:'),
    dcc.RadioItems(id='radio2', #Radio buttons
                  options = [{'label':'Retweet Count', 'value':'retweet_count'},
                             {'label':'Favourite Count', 'value':'favourite_count'}],
                  value='retweet_count'
                  )
    ]),
    html.Div(className='div_graph',
             children=[
    # Graph itself
    dcc.Graph(id='pie_chart')
    ])
    ]),
    
    
    
    # Time series
    html.Div(className='div_container', id='cont2',
             children=[
                     
    html.Aside(className='aside_panel',
             children=[
    html.H3('Sentiment Analysis by Date'),
    html.P('Analysis based on:'),
    dcc.Dropdown(id='ticker',
                  options = [{'label':'Retweet Count', 'value':'retweet_count'},
                             {'label':'Favourite Count', 'value':'favourite_count'},
                             {'label':'Sentiment Rating', 'value':'Vader_sentiment'}],
                  value='favourite_count', clearable=False,
                  style={'width':200}
                  )
        ]),
        html.Div(className='div_graph',
             children=[
        dcc.Graph(id='time-series-chart')
        ])
        ]),
    
     # Heatmap
    html.Div(className='div_container', id='cont3',
             children=[
                     
    html.Aside(className='aside_panel',
             children=[
    html.H3('Most Common Words'),
    html.P('Topic:'),
    dcc.Dropdown(id='heat_dd',
                  options = topics,
                  value="Burren",
                  clearable=False,
                  style={'width':200}
                  )

    ]),
    html.Div(className='div_graph',
             children=[
    dcc.Graph(id='h_map')
    ])
    ]),
     
    
         # Scatter plot
    html.Div(className='div_container', id='cont4',
             children=[
                     
    html.Aside(className='aside_panel',
             children=[
    html.H3('Sentiment by Hour of the Day'),
    html.P('Analysis based on:'),
    dcc.Dropdown(id='scatter_dd1',
                  options = topics,
                  value="Burren",
                  clearable=False,
                  style={'width':200}
                  ),
    
    html.P('Analysis based on:'),
    dcc.Dropdown(id='scatter_dd2',
                  options = ['Vader_sentiment', 'favourite_count'],
                  value="Vader_sentiment",
                  clearable=False,
                  style={'width':200}
                  )
    ]),
    html.Div(className='div_graph',
             children=[
    dcc.Graph(id='scatter')
    ])
    ]),
    
     
     # Geoplot
    html.Div(className='div_container', id='cont5',
             children=[
                     
    html.Aside(className='aside_panel',
             children=[
    html.H3('Location of Tweeter'),
    html.P('Analysis based on:'),
    dcc.RadioItems(id='radio_geo', #Radio buttons
                  options = [{'label':'Retweet Count', 'value':'retweet_count'},
                             {'label':'Favourite Count', 'value':'favourite_count'}],
                  value='retweet_count'
                  )
    ]),
    html.Div(className='div_graph',
             children=[
    dcc.Graph(id='geo_graph') 
     ])
    ])
    ])

# Define callback arguments and function for first graph
@app.callback(
    Output("pie_chart", "figure"),
    Input("radio1", "value"),
    Input("radio2", "value"))

def generate_piechart(radio1, radio2):
    fig = px.pie(df, values=radio2, names=radio1)
    return fig


# Time-series graph
@app.callback(
    Output("time-series-chart", "figure"), 
    Input("ticker", "value"))

def display_time_series(ticker):
    fig = px.area(df, x='created_at', y=ticker)
    
    # Show labels more easily when hovering
    #fig.update_layout(hovermode="x")
    return fig


# Heatmap graph
@app.callback(
    Output("h_map", "figure"), 
    Input("heat_dd", "value"))
    

def generate_heatmap(h_map):
    # Unpack results from function at top
    common_word, common_word_values = get_common_words(h_map)
    
    # Create list for the labels used when hovering
    label_list = []
    labels_lists = []
    for i in range(0, len(common_word)):
        for j in range(0, len(common_word[0])):
            label_list.append("Word:"+ common_word[i][j] +"<br>Frequency:"+ str(common_word_values[i][j]))
        labels_lists = [label_list[x:x+5] for x in range(0, len(label_list), 5)]
    
    fig = go.Figure(data=go.Heatmap(z = common_word_values,
                                    text = common_word,
                                    texttemplate="%{text}",
                                    hovertext = labels_lists
                                    ))
    # Remove meaningless tick labels
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    # Alter hover labels - a little messy still but close
    fig.update_traces(hovertext = labels_lists)
    
    return fig


# Scatter plot
@app.callback(
    Output("scatter", "figure"), 
    Input("scatter_dd1", "value"),
    Input("scatter_dd2", "value"))

def display_scatter(scatter_dd1, scatter_dd2):
    df_topic = df[df['topic'] == scatter_dd1] # Screen df for topic
    sentiment_sum = df_topic.groupby('hour_of_tweet').sum() # Group df by hour
    
    fig = px.line(df, x=sentiment_sum.index.values, y=sentiment_sum[scatter_dd2],
                  markers=True)
    
    # Set fixed range for both axes
    fig.update_layout(xaxis_range=[-0.5,23.5],
                      xaxis= dict(
                          tickmode='linear', # Set ticks for x axis
                          tick0=0,
                          dtick=1,
                          title="Hour of Day"))
    fig.update_layout(yaxis= {'title' : scatter_dd2}) # Title for y axis
    
    # Set markers to change colour and size based on sentiment value
    fig.update_traces(marker=dict(
        color=abs(sentiment_sum[scatter_dd2]),
        size=abs(sentiment_sum[scatter_dd2])
        ), selector=dict(type='scatter'))
    
    # Hover labels needs to be added back in after computations at start of function
    fig.update_traces(hovertemplate='Hour of Tweet: %{x} <br>'+ scatter_dd2 +': %{y}')
    
    # Show labels more easily when hovering
    fig.update_layout(hovermode="x")
    return fig



# Geo graph
@app.callback(
    Output("geo_graph", "figure"), 
    Input("radio_geo", "value"))
    

def generate_geomap(radio_geo):
    loc_group = df.groupby('location').sum()
    fig = px.scatter_geo(loc_group, locations=loc_group.index.values,
                         hover_name=loc_group.index.values, size=radio_geo,
                         color=loc_group.index.values,
                         locationmode="country names",
                         projection="natural earth")
    #Make geoplot bigger...taken from stackoverflow
    fig.update_layout(height=600, margin={"r":30,"t":25,"l":30,"b":30})
    return fig



if __name__ == '__main__':
    app.run_server(debug=True)