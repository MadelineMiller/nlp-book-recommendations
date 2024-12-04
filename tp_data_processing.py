#data_processing
import re
import pandas as pd 
import numpy
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer

def processing_data():
    df = pd.read_csv('books_data.csv.zip')

    processed_df = df[['Title', 'authors', 'publisher', 'publishedDate','infoLink' ,'categories', 'ratingsCount']]

    processed_df.rename(columns={ #edits the names to be simple
        'Title': 'Title',
        'authors': 'Author',
        'publisher': 'Publisher',
        'publishedDate': 'Date',
        'categories': 'Genre',
        'ratingsCount': 'Rating Count'
    }, inplace=True)

    #dropping the NaN values in columns
    processed_df = processed_df.replace(numpy.nan, 0)

    #adding the feature engineering
   # processed_df['Year'] = processed_df['Date'].apply(lambda x: int(str(x)[:4]) if str(x).isdigit() else 0)
    
    

    #genre encoding using one hot to simplify the categories (genre)
    g_processed = processed_df['Genre'].apply(lambda x: re.findall(r'\w+', str(x)) if x != 0 else [])
    mlb = MultiLabelBinarizer()
    g_encode = mlb.fit_transform(g_processed)
    g_df = pd.DataFrame(g_encode, columns=mlb.classes_)
    processed_df = pd.concat([processed_df.reset_index(drop=True), g_df], axis=1)
    

    print(processed_df.columns)
    # print('\n')
    # print(processed_df)
    # print('\n')
    # print(processed_df['Genre'])


processing_data()

    

