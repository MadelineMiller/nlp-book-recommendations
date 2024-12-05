#data_processing
import re
import pandas as pd 
import numpy
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

def processing_data():
    df = pd.read_csv('books_data.csv.zip')

    processed_df = df[['Title', 'description' , 'authors', 'publisher', 'publishedDate','infoLink' ,'categories', 'ratingsCount']]

    #edits the names to be simple
    processed_df.rename(columns={ 
        'Title': 'Title',
        'description': 'Description',
        'authors': 'Author',
        'publisher': 'Publisher',
        'publishedDate': 'Date',
        'infoLink': 'Where to find',
        'categories': 'Genre',
        'ratingsCount': 'Rating'
    }, inplace=True)

    #dropping rows with NaN
    processed_df.dropna(inplace=True)

    #adding the feature engineering
    #processed_df['Year'] = processed_df['Date'].apply(lambda x: int(str(x)[:4]) if str(x).isdigit() else 0)
    
    #genre sorted alphabetically 

    processed_df = processed_df.sort_values(by=['Genre'])
    # print(processed_df.columns)
    # print('\n')
    # print(processed_df)
    # print('\n')
    return processed_df
'''
def get_keywords(): #should get the top words per genre
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3)
    tfidf_matrix = vectorizer.fit_transforrm(group['Description'])
    keywords = vectorizer.get_feature_names_out()
    return ', '.join(keywords)

#need to find a way to store those keywords (possibly new variable or in dataframe?)
out_pd = processing_data()
genre_keywords = out_pd.groupby('Genre').apply(lambda group: get_keywords(group['Description'])).reset_index(name:'Keywords')

working on this but got to go to class

'''
