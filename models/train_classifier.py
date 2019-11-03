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

# plotting function
import matplotlib.pyplot as plt

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


def load_data(database_filepath):    
    """ Returns the data for training the disaster response pipeline. 
    
    Parameters: database_filepath (string) : path to the working directory.
    
    Returns:
        X (numpy array): Tweets.  
        Y (numpy array): Categories associated to each tweet.
        categories (list): list of categories. 
    
    """
        
    engine = create_engine('sqlite:///' + database_filepath )
    df = pd.read_sql_table('Message', engine)
    X = df.message.values    
    Y = df.iloc[:, 4:]     
        
    categories = Y.columns
    categories = categories.tolist()
    
    Y = Y.values 
    
    return X, Y, categories 
    
    


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
    
    

def build_model():
    """ Returns a GridSearchCV object.
    
    Returns: Return a GridSearchCV object to tune a PipeLine from the imbalanced-learn package.
    
    """ 
    
    pipeline = Pipeline([
        ('features', FeatureUnion([
        
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)), 
                ('tfidf', TfidfTransformer())            
            ])), 
        
            ('text_length',  LengthExtractor()) 
                
            ])), 
    
            ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=25, class_weight='balanced', n_jobs=-1)))     
        ])
    
    
    
    # Number of trees in random forest
    #n_estimators = [10 , 50 , 100]
    n_estimators = [100]
    # Maximum number of features 
    max_features = ['sqrt']
    # Maximum number of levels in tree
    #max_depth = [2, 5, 8, 10]
    max_depth = [2]
    
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    
    parameters = {        
        'clf__estimator__n_estimators': n_estimators,
        'clf__estimator__max_features': max_features,
        'clf__estimator__max_depth': max_depth,
        'clf__estimator__bootstrap': bootstrap     
     }

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, scoring=prec_score_multiclass,  cv=3,   verbose=25, n_jobs=-1)    

    return cv 

      
    
def evaluate_model(model, X_test, Y_test, category_names):
    """ Print out the precision, recall, and f1-score for each category.
    
    """
    
    Y_pred = model.predict(X_test)
    
    for col in range(Y_pred.shape[1]):
        print("Category: {}".format(category_names[col]))
        print(classification_report(Y_test[:, col], Y_pred[:, col]))

    


def save_model(model, model_filepath):    
    """ Save the model.
    
    Parameters: model : a model built with the package scikit-learn or imb-learn
                model_filepath: a path to save the model.
    
    Returns: Nothing.
    
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
