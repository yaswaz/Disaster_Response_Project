import json
import plotly
import pandas as pd
import numpy as np
import re

#from nltk.stem import WordNetLemmatizer
#from nltk.tokenize import word_tokenize

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap, Histogram, Box, Scatter
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

#import functions from train_classifier module
import sys
sys.path.insert(0, '/home/workspace/models/')
from train_classifier import tokenize, compute_text_length



# load data
engine = create_engine('sqlite:///../data/DisasterMessages.db')
df = pd.read_sql_table('disaster_df', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# compute length of texts
df['text_length'] = compute_text_length(df['message'])


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extract categories for Heatmap
    category_map = df.iloc[:,4:].corr().values
    category_names = list(df.iloc[:,4:].columns)
    
    


    # extract length of texts
    length_direct = df.loc[df.genre=='direct','text_length']
    length_social = df.loc[df.genre=='social','text_length']
    length_news = df.loc[df.genre=='news','text_length']
    
     # Graph 4
    category_name = df.iloc[:,4:].columns
    category_count = (df.iloc[:,4:] != 0).sum().values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        
        #Graph 1
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        # Graph 2
        {
            'data': [
                Heatmap(
                    x=category_names,
                    y=category_names[::-1],
                    z=category_map
                )    
            ],

            'layout': {
                'title': 'Heatmap of Categories'
            }
        },
        
        # Graph 3
        {
            'data': [
                Box(
                    y=length_direct,
                    name='Direct',
                    opacity=0.5,
                    jitter=0.5
                    #boxpoints='all'
                ),
                Box(
                    y=length_social,
                    name='Social',
                    opacity=0.5
                ),
                Box(
                    y=length_news,
                    name='News',
                    opacity=0.5
                )
            ],

            'layout': {
                'title': 'Distribution of Text Length',
                'yaxis':{
                    'title':'Count',
                    'dtick':'1000',
                    
                },
                'xaxis': {
                    'title':'Text Length'
                }
            }
        },
        
        # Graph 4
        {
            'data': [
                Bar(
                    x=category_name,
                    y=category_count
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35, 'categoryorder':'total descending'
                }
            }
        }

  
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()