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

import nltk
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('vader_lexicon')


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
        'ratingsCount': 'RatingsCount'
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
    # smallest_year = processed_df['Date'].min()

    # tokenizing description to remove stop words (english)
        # what to do with non english stop words?
    stopWords = set(stopwords.words('english'))
    processed_df['Description'] = processed_df['Description'].apply(lambda x: ' '.join([word for word in word_tokenize(str(x)) if word.lower() not in stopWords]))
    processed_df['Description'] = processed_df['Description'].apply(lambda x: ' '.join(x.split())) #removing extra spaces
    
    # unique_genres = set()
    # for genres in processed_df['Genre']:
    #     if isinstance(genres, str):
    #         # Split genres by comma, remove extra spaces and characters
    #         genres_list = [genre.strip("[]'\" ").lower() for genre in genres.split(',')]
    #         unique_genres.update(genres_list)
    #     else:
    #         # Handle non-string genres (optional, depending on your data)
    #         # You can convert them to strings or skip them based on your needs
    #         pass

    # print(f"Unique Genres: {unique_genres}")
    # tokenizing genres
    # implemented MultiLabelBinarizer to encode the genres
    processed_df['Genre'] = processed_df['Genre'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    
    processed_df['Genre'] = processed_df['Genre'].apply(lambda genres: [genre.lower() for genre in genres])


    # normalize ratings to avoid skewness
    processed_df['RatingsCount'] = processed_df['RatingsCount'].apply(lambda x: np.log(x + 1))
    # print(processed_df[['Title', 'RatingsCount']].head())
    
    # dropping rows with NaN after encoding
    processed_df = processed_df.dropna()



    # sending processed data to new csv file to see it
    #processed_df.to_csv('data.csv', index=False)

    # print(processed_df.info())

    # Checking for duplicate columns (for genres)
    # duplicates = processed_df.columns[processed_df.columns.duplicated()]
    # if not duplicates.empty:
    #     print(f"Duplicate columns: {duplicates}")
    # else:
    #     print("No duplicate columns found.")

    return processed_df

# output of processing_data function
out_pd = processing_data()
# print(f"Smallest year: {smallest_year}") --> 1776

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
average_ratings_by_genre = sep_genres.groupby('Genre')['RatingsCount'].mean().reset_index(name='Average Log Rating Count')
genre_words = sep_genres.groupby('Genre')['Description'].apply(lambda group: get_keywords(group, n_words=5)).reset_index(name='Top Keywords')
genre_frame = pd.merge(average_ratings_by_genre, genre_words, on='Genre')
# genre frame has the average log ratings and top keywords for each genre
    # not sure if it will be used in developing model
genre_frame.to_csv('genre_data.csv', index=False)
# print(genre_frame.head())
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
# print("updated out:")
# print(out_pd.head())
# print("columns:")
# print(out_pd.columns)



from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_csv("data.csv")

# Fill missing values in text fields with an empty string
data['Description'] = data['Description'].fillna("")
data['Genre'] = data['Genre'].fillna("")
data['Author'] = data['Author'].fillna("")

# Combine important textual features for similarity computation
data['combined_features'] = (
    data['Description'] + " " + 
    data['Genre'] + " " + 
    data['Author']
)

# Initialize the TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Generate the TF-IDF matrix for the combined features
tfidf_matrix = tfidf.fit_transform(data['combined_features'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# book title, author, decades (instead of # of pages), genres, description

def title_recommendations(title, cosine_sim=cosine_sim, data=data, top_n=10):
    """
    Function to get book recommendations based on a title.
    """
    # Get the index of the book that matches the title
    idx = data[data['Title'].str.lower() == title.lower()].index[0]

    # Get the similarity scores for all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top_n most similar books
    sim_scores = sim_scores[1:top_n+1]

    # Get the indices of the recommended books
    book_indices = [i[0] for i in sim_scores]

    # Return the top_n most similar books
    return data.iloc[book_indices][['Title', 'Author', 'Genre']]


def keyword_recommendations(keywords, tfidf=tfidf, tfidf_matrix=tfidf_matrix, data=data, top_n=10):
    """
    Function to recommend books based on user-inputted keywords.
    """
    # Convert keywords into a TF-IDF vector
    keyword_vector = tfidf.transform([keywords])

    # Compute similarity scores between the keywords and all books
    sim_scores = cosine_similarity(keyword_vector, tfidf_matrix).flatten()

    # Sort books by similarity scores
    sim_indices = sim_scores.argsort()[-top_n:][::-1]

    # Return the top_n recommendations
    return data.iloc[sim_indices][['Title', 'Author', 'Genre']]

# recommendations = title_recommendations("jennings goes to school", top_n=5)
# print("for jennings: ")
# print(recommendations)


# keyword_based = keyword_recommendations("mystery drama", top_n=5)
# print("mystery drama: ")
# print(keyword_based)

# versitile

# for titles
tfidf_title = TfidfVectorizer(stop_words='english')
tfidf_matrix_title = tfidf_title.fit_transform(data['Title'])
cosine_sim_title = cosine_similarity(tfidf_matrix_title, tfidf_matrix_title)

# for authors
tfidf_author = TfidfVectorizer(stop_words='english')
tfidf_matrix_author = tfidf_author.fit_transform(data['Author'])
cosine_sim_author = cosine_similarity(tfidf_matrix_author, tfidf_matrix_author)

# for decades
data['Decades'] = data['Decades'].astype(str).fillna("")
tfidf_decades = TfidfVectorizer()
tfidf_matrix_decades = tfidf_decades.fit_transform(data['Decades'])
cosine_sim_decades = cosine_similarity(tfidf_matrix_decades, tfidf_matrix_decades)

# for genres
tfidf_genre = TfidfVectorizer(stop_words='english')
tfidf_matrix_genre = tfidf_genre.fit_transform(data['Genre'])
cosine_sim_genre = cosine_similarity(tfidf_matrix_genre, tfidf_matrix_genre)
tfidf_description = TfidfVectorizer(stop_words='english')

# for descriptions
tfidf_matrix_description = tfidf_description.fit_transform(data['Description'])
cosine_sim_description = cosine_similarity(tfidf_matrix_description, tfidf_matrix_description)

def get_recommendations(
    title=None, 
    author=None, 
    decades=None, 
    genres=None, 
    description=None, 
    top_n=10, 
    data=data, 
    cosine_sim_title=None, 
    cosine_sim_author=None, 
    cosine_sim_decades=None, 
    cosine_sim_genre=None, 
    cosine_sim_description=None
):
    """
    Function to get book recommendations based on one or multiple features.
    
    Parameters:
        title (str): Book title to base recommendations on (optional).
        author (str): Author to base recommendations on (optional).
        decades (str): Decade to base recommendations on (optional).
        genres (str): Genres to base recommendations on (optional).
        description (str): Description to base recommendations on (optional).
        top_n (int): Number of recommendations to return.
        data (pd.DataFrame): The dataset containing book information.
        cosine_sim_* (numpy.ndarray): Cosine similarity matrices for respective features.
    
    Returns:
        pd.DataFrame: Top N recommended books.
    """
    # initialize an array to store similarity scores for each book
    total_sim_scores = np.zeros(data.shape[0])
    
    # normalize inputs for case insensitivity
    data = data.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
    title = title.lower().strip() if title else None
    author = author.lower().strip() if author else None
    decades = decades.lower().strip() if decades else None
    if genres:
        genres = [genre.lower().strip() for genre in genres]
    else:
        genres = None
    description = description.lower().strip() if description else None
    
    # compute similarity for each feature if provided
    if title and cosine_sim_title is not None:
        idx = data[data['Title'] == title].index
        if not idx.empty:
            idx = idx[0]
            total_sim_scores += cosine_sim_title[idx]
    
    if author and cosine_sim_author is not None:
        idx = data[data['Author'] == author].index
        if not idx.empty:
            idx = idx[0]
            total_sim_scores += cosine_sim_author[idx]
    
    if decades and cosine_sim_decades is not None:
        idx = data[data['Decades'] == decades].index
        if not idx.empty:
            idx = idx[0]
            total_sim_scores += cosine_sim_decades[idx]
    
    if genres and cosine_sim_genre is not None:
        idx = data[data['Genre'].isin(genres)].index
        if not idx.empty:
            idx = idx[0]
            total_sim_scores += cosine_sim_genre[idx]
    
    if description and cosine_sim_description is not None:
        # use TF-IDF to vectorize the input description
        desc_vector = tfidf_description.transform([description])
        sim_scores = cosine_similarity(desc_vector, tfidf_matrix_description).flatten()
        total_sim_scores += sim_scores
    
    # rank the books by total similarity scores
    data['Similarity'] = total_sim_scores
    recommendations = data.sort_values(by='Similarity', ascending=False).head(top_n)
    
    # return the recommended books
    return recommendations[['Title', 'Author', 'Genre', 'Decades', 'Description']].reset_index(drop=True)


#TEST CASES
# print("\n1:\n")
# title_recommendations = get_recommendations(
#     title="jennings goes to school", 
#     top_n=5, 
#     cosine_sim_title=cosine_sim_title
# )
# print(title_recommendations)

# print("\n2:\n")
# multi_feature_recommendations = get_recommendations(
#     title="jennings goes to school",
#     author="anthony buckeridge",
#     genres="children's stories",
#     description="riotous fire practice",
#     top_n=5,
#     cosine_sim_title=cosine_sim_title,
#     cosine_sim_author=cosine_sim_author,
#     cosine_sim_genre=cosine_sim_genre,
#     cosine_sim_description=cosine_sim_description
# )
# print(multi_feature_recommendations)

# print("\n3:\n")
# decades_recommendations = get_recommendations(
#     decades="2000",
#     top_n=5,
#     cosine_sim_decades=cosine_sim_decades
# )
# print(decades_recommendations)

# print("\n4:\n")
# genre_author_recommendations = get_recommendations(
#     genres="children's stories",
#     author="anthony buckeridge",
#     top_n=5,
#     cosine_sim_genre=cosine_sim_genre,
#     cosine_sim_author=cosine_sim_author
# )
# print(genre_author_recommendations)

# print("\n5:\n")
# description_recommendations = get_recommendations(
#     description="A thrilling adventure in a mysterious world",
#     top_n=5,
#     tfidf_description=tfidf_description,
#     tfidf_matrix_description=tfidf_matrix_description
# )
# print(description_recommendations)
