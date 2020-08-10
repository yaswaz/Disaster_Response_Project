import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loads and merges messages and categories to create a single dataframe
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge the two dataframes
    df = messages.merge(categories, how='inner', on='id')
    
    return df

def clean_data(df):
    """
    Takes the categories in the dataframe split the values in the categories column and change to binary 0 or 1
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    category_colnames = row.map(lambda x: x.rstrip('-10'))
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # convert category values to 0 or 1
    for column in categories:
     # set each value to be the last character of the string
        categories[column] = categories[column].str.strip().str[-1]
    
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # replace categories column in df with new columns
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """
     Save clean data into a dataframe in SQL database
    """
    
    engine = create_engine('sqlite:///DisasterMessages.db')
    df.to_sql('disaster_msg_df', engine, index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()