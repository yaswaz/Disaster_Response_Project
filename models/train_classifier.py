import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import warnings

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from scipy.stats import hmean
from scipy.stats.mstats import gmean
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion

def load_data(database_filepath):
    """
    Loads data from SQLite database and split into features and target
    """
    
    #load data from sql database
    engine = create_engine('sqlite:///DisasterMessages.db')
    df = pd.read_sql_table('disaster_msg_df', engine)
    
    # Drop all null entries
    df = df[~(df.isnull().any(axis=1))|((df.original.isnull())&~(df.offer.isnull()))]
    df = df.dropna(subset=['original'])
    
    # Split into features and target
    X = df.message#.values
    Y = df.iloc[:,4:]#.values
    categories = Y.columns
    
    return X, Y, categories


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
    """
    Compute the character length of texts
    """
    return np.array([len(text) for text in data]).reshape(-1, 1)


def build_model():
    """
    Build a model function with a Scikit ML pipeline that process text messages
    according to NLP best-practice and apply a classifier.
    
    """
    # Get the pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([('text', Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                                                 ('tfidf', TfidfTransformer()),
                                                 ])),
                              ('length', Pipeline([('count', FunctionTransformer(compute_text_length, validate=False))]))]
                             )),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))])
    
    # use GridSearch to find optimal parameters to tune model
    parameters = {'features__text__vect__ngram_range': ((1, 1), (1, 2)),
                  'features__text__vect__max_df': (0.75, 1.0)          
             }
    model = GridSearchCV(pipeline, parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """This function applies ML pipeline to a test set and prints out
    model performance (precision, recall and f1score)
    """
    
    # Predict with model
    Y_pred = model.predict(X_test)
    
    # Print precision, recall and f1score
    Y_pred_pd = pd.DataFrame(Y_pred, columns = Y_test.columns)
    for column in Y_test.columns:
        print('------------------------------------------------------\n')
        print('FEATURE: {}\n'.format(column))
        print(classification_report(Y_test[column],Y_pred_pd[column]))
        
    # Print overall accuracy
    overall_accuracy = (Y_pred == Y_test).mean().mean()
    print('Average overall accuracy {0:.2f}% \n'.format(overall_accuracy*100))


def save_model(model, model_filepath):
    """
    Functions saves trained model as pickle file that can be used later
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()