import json
import plotly
import pandas as pd


from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Histogram
from plotly.graph_objs import Box


import sys

# import nlp libraries
from sqlalchemy import create_engine 
import numpy as np
import pandas as pd 
import string 
import re 
import nltk
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# scikit-learn libraries 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score,  make_scorer


# to save and load scikit-learn models
import pickle 

# custom scoring function for multioutput classifier with imbalaced dataset 
prec_score_multiclass =  make_scorer(precision_score, average='weighted') 


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization

# In order to process the data we adapt code from the following web site: 
# https://blog.chapagain.com.np/python-nltk-twitter-sentiment-analysis-natural-language-processing-nlp/


from nltk.corpus import stopwords 
stopwords_english = stopwords.words('english')
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer


# Happy Emoticons
emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
 
# Sad Emoticons
emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
 
# all emoticons (happy + sad)
emoticons = emoticons_happy.union(emoticons_sad)




app = Flask(__name__)

def tokenize(text):
    """ Returns a tokenized tweet. This functions do the following taks: 
        Remove retweet text "RT"
        Remove hyperlinks
        Remove hashtags (only the hashtag # and not the word)
        Remove stop words like a, and, the, is, are, etc.
        Remove emoticons like :), :D, :(, :-), etc.
        Remove punctuation like full-stop, comma, exclamation sign, etc.
        Apply lemmatization first and then stemming.
        
    Parameters: text (string): tweet data.     

    Returns: words (list of strings): tokenized tweet.    
    
    """    
    
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', text)
 
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    
    tweet_tokens = tokenizer.tokenize(tweet)
    
    # removing words that are common
    words = [w for w in tweet_tokens if w not in stopwords.words("english")]
    
    # removing emoticons 
    words = [w for w in words if w not in emoticons]
        
    # removing punctuations 
    words = [ re.sub(r'[^\w\s]', '', w) for w in words]
    
    # removing words that are equal to ''
    words = [ w for w in words if w != '']
    
    # removing words that has numbers 
    words = [w for w in words if w.isalpha() ]

    # removing words that has length equal to 1 
    words = [w for w in words if len(w) > 1 ]
            
    # we apply lemmatization first and then stemming
    # we follow a suggestion from a video that is part of 
    # Data Scientist Nanodegree from Udacity
    words = [WordNetLemmatizer().lemmatize(w) for w in words ]
    words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words ]
    
    words = [PorterStemmer().stem(w) for w in words ]
    
    
    return words     

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Message', engine)


class LengthExtractor(BaseEstimator, TransformerMixin):
    
    """ This class is a transformer that will be used in a Pipeline to extract the lenght of the tweets.
    
    """
        
    def fit(self, X, y=None):
        """ Fit method 
    
        Parameters:
            self: the self object. 
            X : input variables
            y : output variables
    
        Returns: 
            self: the self object. 
    """         
        
        return self

    def transform(self, X):
        """ Transform method 
    
        Parameters:
            self: the self object. 
            X : input variables
    
        Returns: 
            A 2d-numpy array containing the length of the tweets. 
    """                 
        
        return pd.Series(X).apply(lambda x: len(x)).values[:, None]



# load model
model = pickle.load(open("../models/classifier.pkl", 'rb'))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    length_tweets = df["message"].apply(lambda x: len(x))
    
    
    
    # create visuals
    
    graph_one = []
    
    graph_one.append(
           Bar(
                  x=genre_names,
                  y=genre_counts
              )                
    )
    
    layout_one = dict(title = 'Distribution of Message Genres',
                      xaxis = dict(title = "Genre"),
                      yaxis = dict(title = "Count")        
    )
    
    
    graph_two = []
    
    graph_two.append(
           Histogram(
                  x=length_tweets                 
              )                
    )
    

    layout_two = dict(title = 'Distribution of length of the tweets',
                      xaxis = dict(title = "Lenght of the tweets"),
                      yaxis = dict(title = "Frecuency")        
    )
        
    
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(figures)]
    graphJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
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
