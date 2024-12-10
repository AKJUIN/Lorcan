pip install pandas scikit-learn flask
import pandas as pd

# Load the dataset
books = pd.read_csv('books.csv')

# Display basic information
print(books.head())

# Focus on useful columns
books = books[['title', 'authors', 'average_rating', 'genres']]
books['genres'] = books['genres'].fillna('')
# Handle missing genres
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create a TF-IDF matrix for genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(books['genres'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend books
def recommend_books(book_title, num_recommendations=5):
    # Get the index of the book
    idx = books[books['title'].str.contains(book_title, case=False, na=False)].index[0]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top recommendations
    sim_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]

    # Return book titles
    return books.iloc[sim_indices][['title', 'authors', 'average_rating']]
