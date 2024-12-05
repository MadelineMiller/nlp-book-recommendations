#data_processing
import re
import pandas as pd 
import numpy as np
import sklearn
import ast
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
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
    
    #clean titles
    processed_df['Title'] = processed_df['Title'].str.strip().str.lower()

    #genre sorted alphabetically 
    processed_df = processed_df.sort_values(by=['Genre'])

    # process publication years to add that to search instead of book length
    processed_df['Date'] = pd.to_datetime(processed_df['Date'], errors='coerce').dt.year
    processed_df = processed_df.dropna(subset=['Date'])
    processed_df['Date'] = processed_df['Date'].astype('Int64')

    # tokenizing description to remove stop words (english)
        # what to do with non english stop words?
    stopWords = set(stopwords.words('english'))
    processed_df['Description'] = processed_df['Description'].apply(lambda x: ' '.join([word for word in word_tokenize(str(x)) if word.lower() not in stopWords]))
    processed_df['Description'] = processed_df['Description'].apply(lambda x: ' '.join(x.split())) #removing extra spaces
   
    # tokenizing genres
    # implemented MultiLabelBinarizer to encode the genres
    processed_df['Genre'] = processed_df['Genre'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    processed_df['Genre'] = processed_df['Genre'].apply(lambda genres: [genre.lower() for genre in genres])


    # normalize ratings to avoid skewness
    processed_df['Rating'] = processed_df['Rating'].apply(lambda x: np.log(x + 1))
    print(processed_df[['Title', 'Rating']].head())
    
    # dropping rows with NaN after encoding
    processed_df = processed_df.dropna()



    # sending processed data to new csv file to see it
    #processed_df.to_csv('data.csv', index=False)

    print(processed_df.info())

    # Checking for duplicate columns (for genres)
    # duplicates = processed_df.columns[processed_df.columns.duplicated()]
    # if not duplicates.empty:
    #     print(f"Duplicate columns: {duplicates}")
    # else:
    #     print("No duplicate columns found.")

    return processed_df

# output of processing_data function
out_pd = processing_data()

################### Feature Engineering ####################


#**************** Genre Specific Features **************** 
# separate dataframe for genre specific features: top keywords and average log ratings by genre
ungrp_genres = out_pd.explode('Genre')
single_genres = ungrp_genres['Genre'].value_counts()
max_val = 5
high_genres = single_genres[single_genres >= max_val].index

# filter out genres that have less than 5 books and delete any rows with no genres
out_pd['Genre'] = out_pd['Genre'].apply(lambda genres: [genre for genre in genres if genre in high_genres])
out_pd = out_pd[out_pd['Genre'].str.len() > 0]

# now encode the genres
mlb = MultiLabelBinarizer()
genre_encodings = mlb.fit_transform(out_pd['Genre'])
genre_df = pd.DataFrame(genre_encodings, columns=mlb.classes_)
out_pd = pd.concat([out_pd.reset_index(drop=True), genre_df], axis=1)
    # genres are now encoded as binary columns : 1 if book is in that genre, 0 otherwise


def get_keywords(input, n_words=5): #should get the top words per genre
    vectorizer = TfidfVectorizer(stop_words='english', max_features=n_words)
    tfidf_matrix = vectorizer.fit_transform(input)
    keywords = vectorizer.get_feature_names_out()
    return ', '.join(keywords)

sep_genres = out_pd.explode('Genre')

# separate dataframe for genre specific features: top keywords and average log ratings by genre
average_ratings_by_genre = sep_genres.groupby('Genre')['Rating'].mean().reset_index(name='Average Log Rating')
genre_words = sep_genres.groupby('Genre')['Description'].apply(lambda group: get_keywords(group, n_words=5)).reset_index(name='Top Keywords')
genre_frame = pd.merge(average_ratings_by_genre, genre_words, on='Genre')
# genre frame has the average log ratings and top keywords for each genre
    # not sure if it will be used in developing model
genre_frame.to_csv('genre_data.csv', index=False)
print(genre_frame.head())
#**************** Genre Specific Features **************** 



# add features to out_pd by doing sentiment analysis on descriptions and also review summaries and full reviews
sentiment_agent = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sentiment_agent.polarity_scores(str(text))
    return score['compound']


# Description sentiment
out_pd['Description_Sentiment'] = out_pd['Description'].apply(get_sentiment)

# add feature for publication decade
out_pd['Decades'] = (out_pd['Date'] // 10) * 10

# add feature for if book has multiple authors
    # if text has more than one contributor, it is more likely to be educational or academic in some way?
out_pd['NumberContributors'] = out_pd['Author'].apply(lambda x: len(str(x).split(',')))

out_pd.to_csv('data.csv', index=False)
print("updated out:")
print(out_pd.head())
print("columns:")
print(out_pd.columns)


# #data_processing
# import re
# import pandas as pd 
# import numpy as np
# import sklearn
# import ast
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.feature_extraction.text import TfidfVectorizer

# def processing_data():
#     # Read and Load the data
#     df = pd.read_csv('books_data.csv.zip')

#     # Give the columns names
#     processed_df = df[['Title', 'description' , 'authors', 'publisher', 'publishedDate','infoLink' ,'categories', 'ratingsCount']]

#     #edits the names to be simple
#     processed_df = processed_df.rename(columns={ 
#         'Title': 'Title',
#         'description': 'Description',
#         'authors': 'Author',
#         'publisher': 'Publisher',
#         'publishedDate': 'Date',
#         'infoLink': 'Where to find',
#         'categories': 'Genre',
#         'ratingsCount': 'Rating'
#     })
  
#     #dropping rows with NaN initially
#     processed_df = processed_df.dropna()
    
#     #genre sorted alphabetically 
#     processed_df = processed_df.sort_values(by=['Genre'])

#     # tokenizing description to remove stop words (english)
#         # what to do with non english stop words?
#     stopWords = set(stopwords.words('english'))
#     processed_df['Description'] = processed_df['Description'].apply(lambda x: ' '.join([word for word in word_tokenize(str(x)) if word.lower() not in stopWords]))
#     processed_df['Description'] = processed_df['Description'].apply(lambda x: ' '.join(x.split())) #removing extra spaces
   
#     # tokenizing genres
#     # implemented MultiLabelBinarizer to encode the genres
#     processed_df['Genre'] = processed_df['Genre'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
#     processed_df['Genre'] = processed_df['Genre'].apply(lambda genres: [genre.lower() for genre in genres])
#     mlb = MultiLabelBinarizer()
#     genre_encodings = mlb.fit_transform(processed_df['Genre'])
#     genre_df = pd.DataFrame(genre_encodings, columns=mlb.classes_)
#     processed_df = pd.concat([processed_df, genre_df], axis=1)

#     # normalize ratings to avoid skewness
#     processed_df['Rating'] = processed_df['Rating'].apply(lambda x: np.log(x + 1))
#     print(processed_df[['Title', 'Rating']].head())
    
#     # dropping rows with NaN after encoding
#     processed_df = processed_df.dropna()

#     #print("Processed Data")
#     #print(processed_df.columns)

#     # sending processed data to new csv file to see it
#     processed_df.to_csv('data.txt', sep='\t', index=False) 

#     # Checking for duplicate columns (for genres)
#     # duplicates = processed_df.columns[processed_df.columns.duplicated()]
#     # if not duplicates.empty:
#     #     print(f"Duplicate columns: {duplicates}")
#     # else:
#     #     print("No duplicate columns found.")

#     return processed_df

# data = processing_data()


# '''
# def get_keywords(): #should get the top words per genre
#     vectorizer = TfidfVectorizer(stop_words='english', max_features=3)
#     tfidf_matrix = vectorizer.fit_transforrm(group['Description'])
#     keywords = vectorizer.get_feature_names_out()
#     return ', '.join(keywords)

# #need to find a way to store those keywords (possibly new variable or in dataframe?)
# out_pd = processing_data()
# genre_keywords = out_pd.groupby('Genre').apply(lambda group: get_keywords(group['Description'])).reset_index(name:'Keywords')

# working on this but got to go to class

# '''
