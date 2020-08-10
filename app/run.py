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
from plotly.graph_objs import Bar, Heatmap, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    Remove special characters and capital letters and lemmatize texts
    """
    # Identify and replace url with placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9_]", " ", text)
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # Remove stop words
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token and lemmatize, normalize case, and remove leading/trailing white space
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def compute_text_length(data):
    return np.array([len(text) for text in data]).reshape(-1, 1)

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
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extract categories
    category_map = df.iloc[:,4:].corr().values
    category_names = list(df.iloc[:,4:].columns)

    # extract length of texts
    length_direct = df.loc[df.genre=='direct','text_length']
    length_social = df.loc[df.genre=='social','text_length']
    length_news = df.loc[df.genre=='news','text_length']
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
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
        
        # Graph Two
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