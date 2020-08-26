### Date created
20th July 2020

### Project Title
Disaster Response Pipeline Project

### Description

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. Figure Eight provides the data to build a model for an API that classifies disaster messages.
When disaster strikes, it creates different and complex challenges for the organizations tasked with the responsibility of responding. They have to go through thousands of messages to understand what is needed in every situation. Filtering out these messages and deciding which are need their response is a daunting task especially in large disaster situations. In this project predictive modeling is used to make the classification of these messages easier and more efficient.

The project is divided into 3 sections:
1. ETL Pipeline -  The first part of the data pipeline extracts, transforms, and loads the messages and categories data. The datasets are cleaned and then stored in a SQLite database. All the data cleaning is done with pandas. The ETL script is stored as process_data.py

2. Machine Learning Pipeline - For the machine learning section,  the data is split into a training set and a test set. Then, a machine learning pipeline is created that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). The text data was trained on a multioutput classifier model using AdaBoost Classifier.  Finally, the model is exported as a pickle file which is then used in the machine learning script saved as train_classifier.py. The AdaBoost classifier model scored had an overall accuracy of 92.88% after tuning the parameters using GridSearchCV

3. Web App - This shows the model results in real time. The Flask app classifies input messages and shows visualizations of key statistics of the dataset.

### INSTALATION
- Python 3.5 and above
- Machine Learning Libraries: NumPy, scipy, Pandas, Scikit-Learn
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraries: SQLalchemy
- Web App and Data Visualization: Flask, Plotly

### Files

- process_data.py
- train_classifier.py
- run.py
- ETL Pipeline Preparation.ipynb
- ML Pipeline Preparation.ipynb


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Author
- Yasir Waziri


### Acknowledgements

- Udacity Data Scientist Nanodegree that provided the platform for the project
- Figure Eight for providing messages dataset
- https://github.com/matteobonanomi/dsnd-disaster-response
- https://github.com/kzhao682/Disaster_Response
- stackoverflow.com

### Screenshots
