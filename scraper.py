import requests, json, datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
from nltk.corpus import stopwords
from string import punctuation
import streamlit as st
from apify_client import ApifyClient

def gt(dt_str):
     '''
     Converts an isoformat string to a datetime object.
     '''
     dt, _, us = dt_str.partition(".")
     dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
     us = int(us.rstrip("Z"), 10)
     return dt + datetime.timedelta(microseconds=us)

def query_for_usage():
    '''
    Queries Apify for the number of queries used.
    '''
    print('Querying for usage...')
    url = 'https://api.apify.com/v2/users/me/usage/monthly?token=' + st.secrets["APIFY_TOKEN"]
    r = requests.get(url)
    d = json.loads(r.text)
    date = gt(d['data']['usageCycle']['endAt'])
    date_diff = (date - datetime.datetime.now())
    return f"**{d['data']['monthlyServiceUsage']['PROXY_SERPS']['quantity']*100} / 50,000** queries used this month. Resets in **{date_diff.days} days, {date_diff.seconds//3600} hours**."

@st.cache_data
def query_google(query: str, num_of_queries: int, use_json=True):
    '''
    Queries Google for a given query.
    Returns a list of descriptions and a list of ratings.
    '''
    print(f'Searching for {num_of_queries} reviews...')
    if use_json:
        with open('data.json') as f:
            res_ls = json.load(f)
    else:
        client = ApifyClient(st.secrets["APIFY_TOKEN"])

        run_input = { "queries": f"{query} review", 
                    "maxPagesPerQuery": num_of_queries // 100,
                    "resultsPerPage": 100,
                    "countryCode": "",
                    "customDataFunction": """async ({ input, $, request, response, html }) => {
                    return {
                    pageTitle: $('title').text(),
                    };
                };""",
                }
        run = client.actor("apify/google-search-scraper").call(run_input=run_input)
        res_ls = []
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            res_ls.append(item)
    

    rating_dataset = []
    desc_dataset = []
    for res in res_ls:
        # res is a dictionary
        for row in res['organicResults']:
            try:
                rating = float(row['productInfo']['rating'])
                if rating > 10:
                    rating = rating/100
                elif rating > 5:
                    rating = rating/10
                else:
                    rating = rating/5
                rating_dataset.append(rating)
            except:
                pass

            desc = row['description'].replace('\xa0','')
            if len(desc) > 14:

                # Check case for Nov XX, XXXX
                if desc[12].isalpha() and desc[8:12].isdigit():
                    desc = desc[12:]

                # Check case for Nov X, XXXX
                elif desc[11].isalpha() and desc[7:11].isdigit():
                    desc = desc[11:]

                desc_dataset.append(desc)
    

    return desc_dataset, np.array(rating_dataset, dtype=np.float32)


def create_wordcloud(desc_dataset: list):
    '''
    Creates a wordcloud from a list of descriptions.
    '''
    pos_words = ''
    neg_words = ''
    neutral_words = ''
    stop_words = set(stopwords.words('english'))

    for desc in desc_dataset:
        for word in desc.split():
            word = word.lstrip(punctuation).rstrip(punctuation)
            if word:
                analysis = TextBlob(word)
                if analysis.sentiment.polarity > 0:
                    pos_words += word + ' '
                elif analysis.sentiment.polarity < 0:
                    neg_words += word + ' '
                else:
                    neutral_words += word + ' '

    figs = []
    for title, words in [('Positive words', pos_words), ('Negative words', neg_words), ('Neutral words', neutral_words)]:
        wordcloud = WordCloud(background_color ='white',
                        stopwords = stop_words,
                        min_font_size = 10,
                        max_words = 20).generate(words)
        
        # Plot the WordCloud image
        fig, ax = plt.subplots(figsize=(15,5))
        ax.set_facecolor('black')
        plt.imshow(wordcloud)
        plt.title(title, fontsize=55, color='black', pad=40)
        plt.axis("off")
        plt.show()
        figs.append(fig)
    return figs



def show_ratings(rating_dataset, rating_round=10, plot_type='line'):
    '''
    Shows a chart of the pure ratings.
    rating_round: 0 = Continuous (0-1), 5 = Integer /5, 10 = Integer /10
    plot_type: 'Line', 'Scatter', 'Both' <- For continuous only
    '''
    fig, ax = plt.subplots(figsize=(6,6))
    if rating_round == 5 or rating_round == 10:
        if rating_round == 5:
            rating_dataset = np.vectorize(lambda x: round(x * 5))(rating_dataset)
        else:
            rating_dataset = np.vectorize(lambda x: round(x * 10))(rating_dataset)
        
        # Count occurrences of each loan
        df = pd.Series(rating_dataset)
        df2 = df.value_counts()
        df2 = df2.reindex(list(range(1, rating_round + 1)))
        df2.plot(ax=ax, kind='bar')

    else:
        x ,y  = np.unique(rating_dataset, return_counts=True) 
        if plot_type == 'Line':
            plt.plot(x,y)
        elif plot_type == 'Scatter':
            plt.scatter(x,y)
        elif plot_type == 'Both':
            plt.plot(x,y)
            plt.scatter(x,y)
        plt.xlim(xmin=0, xmax=1)
    plt.title('Raw Ratings', fontsize=15, color='black')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()
    return fig


def eval_sentiment(desc_dataset, model, row_labels, row_values, title):
    '''
    Shows a chart of the sentiment of the descriptions.
    '''
    res = model(desc_dataset)
    df = pd.DataFrame(res)['label'].str.capitalize()
    fig, ax = plt.subplots(figsize=(8,8))
    df2 = df.value_counts()
    df2 = df2.reindex(row_labels)
    df2.plot(ax=ax, kind='bar')
    plt.title(title, fontsize=25, color='black')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()

    total = 0
    count = 0
    total = 0
    for df2_val, row_val in zip(df2, row_values):
        if not pd.isna(df2_val):
            total += row_val * df2_val
            count += df2_val

    mean = total / len(desc_dataset)
        
    
    return fig, mean