#data_processing
import re
import pandas as pd 
import numpy as np
import sklearn
import ast
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

def processing_data():
    # Read and Load the data
    df = pd.read_csv('books_data.csv.zip')

    # Give the columns names
    processed_df = df[['Title', 'description' , 'authors', 'publisher', 'publishedDate','infoLink' ,'categories', 'ratingsCount']]

    #edits the names to be simple
    processed_df = processed_df.rename(columns={ 
        'Title': 'Title',
        'description': 'Description',
        'authors': 'Author',
        'publisher': 'Publisher',
        'publishedDate': 'Date',
        'infoLink': 'Where to find',
        'categories': 'Genre',
        'ratingsCount': 'Rating'
    })
  
    #dropping rows with NaN initially
    processed_df = processed_df.dropna()
    
    #genre sorted alphabetically 
    processed_df = processed_df.sort_values(by=['Genre'])

    # tokenizing description to remove stop words (english)
        # what to do with non english stop words?
    stopWords = set(stopwords.words('english'))
    processed_df['Description'] = processed_df['Description'].apply(lambda x: ' '.join([word for word in word_tokenize(str(x)) if word.lower() not in stopWords]))
    processed_df['Description'] = processed_df['Description'].apply(lambda x: ' '.join(x.split())) #removing extra spaces
   
    # tokenizing genres
    # implemented MultiLabelBinarizer to encode the genres
    processed_df['Genre'] = processed_df['Genre'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    processed_df['Genre'] = processed_df['Genre'].apply(lambda genres: [genre.lower() for genre in genres])
    mlb = MultiLabelBinarizer()
    genre_encodings = mlb.fit_transform(processed_df['Genre'])
    genre_df = pd.DataFrame(genre_encodings, columns=mlb.classes_)
    processed_df = pd.concat([processed_df, genre_df], axis=1)

    # normalize ratings to avoid skewness
    processed_df['Rating'] = processed_df['Rating'].apply(lambda x: np.log(x + 1))
    print(processed_df[['Title', 'Rating']].head())
    
    # dropping rows with NaN after encoding
    processed_df = processed_df.dropna()

    #print("Processed Data")
    #print(processed_df.columns)

    # sending processed data to new csv file to see it
    processed_df.to_csv('data.txt', sep='\t', index=False) 

    # Checking for duplicate columns (for genres)
    # duplicates = processed_df.columns[processed_df.columns.duplicated()]
    # if not duplicates.empty:
    #     print(f"Duplicate columns: {duplicates}")
    # else:
    #     print("No duplicate columns found.")

    return processed_df

data = processing_data()


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
