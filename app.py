import pandas as pd
from flask import Flask, request, jsonify
from tp_data_processing import get_recommendations, data, cosine_sim_title, cosine_sim_author, cosine_sim_decades, cosine_sim_genre, cosine_sim_description
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This allows all origins by default


@app.route('/recommend', methods=['GET'])
def recommend_books():
    title = request.args.get('title', default=None, type=str)
    author = request.args.get('author', default=None, type=str)
    decades = request.args.get('decades', default=None, type=str)
    genres = request.args.get('genres', default=None, type=str)
    description = request.args.get('description', default=None, type=str)
    top_n = request.args.get('top_n', default=5, type=int)

    # Fetch recommendations
    recommendations = get_recommendations(
        title=title,
        author=author,
        decades=decades,
        genres=genres,
        description=description,
        top_n=top_n,
        data=data,
        cosine_sim_title=cosine_sim_title,
        cosine_sim_author=cosine_sim_author,
        cosine_sim_decades=cosine_sim_decades,
        cosine_sim_genre=cosine_sim_genre,
        cosine_sim_description=cosine_sim_description
    )

    # Clean and format recommendations
    cleaned_recommendations = recommendations.apply(
        lambda x: {
            "Title": x['Title'],
            "Author": [x['Author']] if isinstance(x['Author'], str) else x['Author'],
            "Genre": x['Genre'].split(',') if isinstance(x['Genre'], str) else x['Genre'],
            "Decades": x['Decades'],
            "Description": x['Description']
        }, axis=1
    )

    # Convert to JSON format
    return jsonify(cleaned_recommendations.tolist())

if __name__ == '__main__':
    app.run(debug=True)
