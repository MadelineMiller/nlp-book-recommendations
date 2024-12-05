#data_processing
import re
import pandas as pd 
import numpy
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder

def processing_data():
    df = pd.read_csv('books_data.csv.zip')

    processed_df = df[['Title', 'authors', 'publisher', 'publishedDate','infoLink' ,'categories', 'ratingsCount']]

    #edits the names to be simple
    processed_df.rename(columns={ 
        'Title': 'Title',
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

    processed_df = processed_df.sort_values(by='Genre')

    # #genre encoding using one hot to simplify the categories (genre)
    # processed_df = processed_df.copy()


    # g_processed = processed_df['Genre'].apply(lambda x: re.findall(r'\w+', str(x)) if pd.notnull else [])
    # mlb = MultiLabelBinarizer()
    # g_encode = mlb.fit_transform(g_processed)
    # g_df = pd.DataFrame(g_encode, columns=mlb.classes_)
    # processed_df = pd.concat([processed_df.reset_index(drop=True), g_df], axis=1)
    

    print(processed_df.columns)
    print('\n')
    print(processed_df)
    print('\n')
    print(processed_df['Genre'])



processing_data()

    

