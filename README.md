# Disaster Response Pipeline Project from Data Scientist Nanodegree

## Introduction 
In this project we have used a machine learning pipeline 
to categorize disaster events in real messages. The data was 
provided by Figure Eight.

## The data 
The data consits of 26109 tweets classified in 36 categories.

## Project components


### ETL Pipeline
In the Python script [process_data.py](https://github.com/ricardoues/disaster_response_pipeline/blob/master/data/process_data.py), we wrote a data pipeline 
that cleans the data.

### ML Pipeline
In the Python script [train_classifier.py](https://github.com/ricardoues/disaster_response_pipeline/blob/master/models/train_classifier.py), we wrote a machine 
learning pipeline that train a random forest model and save it
into a pickle file.

### Flask Web App 
In the Python script [run.py](https://github.com/ricardoues/disaster_response_pipeline/blob/master/app/run.py), we deploy the machine learning 
model in a web app written in the Flask web framework. 

### Dependencies 
In order to run the code I suggest you to have a machine with at least 8 CPU cores, moreover you should have Python 3.5.2. 

### How to run 
1. Clone the repository and run the following commands in the project's root directory to set up your database and model.

    - To run the Python packages dependencies
         `pip install -r /path/to/requirements.txt`
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


There is a video that shows the web app in the following link:

[Disaster Response Pipeline video](https://www.youtube.com/watch?v=jnPsk7x53lo)

### References

[TF-IDF pipeline](https://medium.com/@chrisfotache/text-classification-in-python-pipelines-nlp-nltk-tf-idf-xgboost-and-more-b83451a327e0)
[RandomizedSearchCV](https://medium.com/rants-on-machine-learning/smarter-parameter-sweeps-or-why-grid-search-is-plain-stupid-c17d97a0e881)
[Plotly reference](https://plot.ly/python/reference/)
[Deploying a Flask app to heroku](https://stackabuse.com/deploying-a-flask-application-to-heroku/)
