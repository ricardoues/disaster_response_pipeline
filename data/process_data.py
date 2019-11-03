import sys
import pandas as pd 
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ Returns a pandas dataframe containing information about tweets and categories. 
    
    Parameters: messages_filepath (string) : path to the messages file.
                categories_filepath (string) : path to the categories file.
    
    Returns:
        df (pandas dataframe): A pandas dataframe with the information of tweets and categories.  
    
    """
    
    
    
    # Read the messages and categories files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge the dataframes: messages and categories
    df = pd.merge(messages, categories, on='id')
    
    return df 
    
    
def clean_data(df):
    """ Returns a clean pandas dataframe. 
    
    Parameters: df (string) : unclean 
    
    Returns:
        df (pandas dataframe): A clean pandas dataframe.  
    
    """
       
    
    # Splitting categories 
    categories = df.categories.str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    # converting category values to just numbers 0 or 1 
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(x))
        
    # replacing  categories column in df with new categories column 
    
    df = df.drop(['categories'], axis=1)    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # removing duplicates 
    
    df.drop_duplicates(subset="message", keep=False, inplace=True)
    
    # the variable related has three categories according to the following forum post
    # https://knowledge.udacity.com/questions/53182
    # that is why we will replace the value 2 with the most frequent value.
    df['related'] = df['related'].replace(2, 1)
    
    return df 
           


def save_data(df, database_filename):
    """ Returns a clean pandas dataframe. 
    
    Parameters: df (string) : unclean 
    
    Returns:
        df (pandas dataframe): A clean pandas dataframe.  
    
    """
    
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Message', engine, index=False)


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
