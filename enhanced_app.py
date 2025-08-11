from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Global variables to store loaded models
models = {}
is_loaded = False

def load_models():
    """Load all the trained models and data"""
    global models, is_loaded
    
    try:
        model_files = {
            'books_clean': '../models/books_clean.pkl',
            'ratings_filtered': '../models/ratings_filtered.pkl',
            'user_item_matrix': '../models/user_item_matrix.pkl',
            'item_similarity_df': '../models/item_similarity_df.pkl',
            'user_to_idx': '../models/user_to_idx.pkl',
            'idx_to_user': '../models/idx_to_user.pkl',
            'book_to_idx': '../models/book_to_idx.pkl',
            'idx_to_book': '../models/idx_to_book.pkl',
            'svd_model': '../models/svd_model.pkl',
            'user_factors': '../models/user_factors.pkl',
            'item_factors': '../models/item_factors.pkl',
            'popular_books': '../models/popular_books.pkl'
        }
        
        for model_name, file_path in model_files.items():
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
                print(f"✓ Loaded {model_name}")
            else:
                print(f"✗ File not found: {file_path}")
        
        is_loaded = True
        print("✅ All models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        is_loaded = False

# Load models at startup
load_models()

# ===================================================================
# RECOMMENDATION FUNCTIONS
# ===================================================================

def get_similar_books_api(isbn, n_recommendations=10):
    """Get similar books using item-based collaborative filtering"""
    try:
        if not is_loaded:
            return {"error": "Models not loaded"}
        
        item_similarity_df = models['item_similarity_df']
        books_clean = models['books_clean']
        
        if isbn not in item_similarity_df.index:
            return {"error": f"Book with ISBN {isbn} not found"}
        
        # Get similarity scores
        similar_scores = item_similarity_df[isbn].sort_values(ascending=False)
        similar_books = similar_scores[1:n_recommendations+1]
        
        recommendations = []
        for book_isbn, similarity_score in similar_books.items():
            book_info = books_clean[books_clean['ISBN'] == book_isbn]
            if not book_info.empty:
                recommendations.append({
                    'isbn': book_isbn,
                    'title': book_info['Book-Title'].iloc[0],
                    'author': book_info['Book-Author'].iloc[0],
                    'year': int(book_info['Year-Of-Publication'].iloc[0]),
                    'publisher': book_info['Publisher'].iloc[0] if pd.notna(book_info['Publisher'].iloc[0]) else "Unknown",
                    'similarity_score': float(similarity_score),
                    'image_url': book_info['Image-URL-M'].iloc[0] if pd.notna(book_info['Image-URL-M'].iloc[0]) else ""
                })
        
        return {"recommendations": recommendations}
        
    except Exception as e:
        return {"error": str(e)}

def get_user_recommendations_api(user_id, n_recommendations=10):
    """Get personalized recommendations for a user"""
    try:
        if not is_loaded:
            return {"error": "Models not loaded"}
        
        user_item_matrix = models['user_item_matrix']
        item_similarity_df = models['item_similarity_df']
        books_clean = models['books_clean']
        
        if user_id not in user_item_matrix.index:
            return {"error": f"User {user_id} not found"}
        
        # Get user's ratings
        user_ratings = user_item_matrix.loc[user_id]
        rated_books = user_ratings[user_ratings > 0]
        
        if len(rated_books) == 0:
            return {"error": "User has not rated any books"}
        
        # Calculate recommendations
        recommendations = {}
        
        for book_isbn, rating in rated_books.items():
            if book_isbn in item_similarity_df.index:
                similar_books = item_similarity_df[book_isbn].sort_values(ascending=False)
                
                for sim_book_isbn, similarity in similar_books.items():
                    if sim_book_isbn not in rated_books.index and similarity > 0.1:
                        if sim_book_isbn in recommendations:
                            recommendations[sim_book_isbn] += rating * similarity
                        else:
                            recommendations[sim_book_isbn] = rating * similarity
        
        # Sort and format recommendations
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        final_recommendations = []
        for isbn, score in sorted_recommendations[:n_recommendations]:
            book_info = books_clean[books_clean['ISBN'] == isbn]
            if not book_info.empty:
                final_recommendations.append({
                    'isbn': isbn,
                    'title': book_info['Book-Title'].iloc[0],
                    'author': book_info['Book-Author'].iloc[0],
                    'year': int(book_info['Year-Of-Publication'].iloc[0]),
                    'publisher': book_info['Publisher'].iloc[0] if pd.notna(book_info['Publisher'].iloc[0]) else "Unknown",
                    'recommendation_score': float(score),
                    'image_url': book_info['Image-URL-M'].iloc[0] if pd.notna(book_info['Image-URL-M'].iloc[0]) else ""
                })
        
        return {"recommendations": final_recommendations}
        
    except Exception as e:
        return {"error": str(e)}

def get_svd_recommendations_api(user_id, n_recommendations=10):
    """Get recommendations using SVD matrix factorization"""
    try:
        if not is_loaded:
            return {"error": "Models not loaded"}
        
        user_to_idx = models['user_to_idx']
        idx_to_book = models['idx_to_book']
        user_item_matrix = models['user_item_matrix']
        user_factors = models['user_factors']
        item_factors = models['item_factors']
        books_clean = models['books_clean']
        
        if user_id not in user_to_idx:
            return {"error": f"User {user_id} not found"}
        
        user_idx = user_to_idx[user_id]
        user_ratings = user_item_matrix.loc[user_id]
        
        # Predict ratings
        predicted_ratings = np.dot(user_factors[user_idx], item_factors.T)
        
        # Create recommendations
        recommendations = []
        for i, predicted_rating in enumerate(predicted_ratings):
            book_isbn = idx_to_book[i]
            if user_ratings[book_isbn] == 0:  # Not rated by user
                recommendations.append((book_isbn, predicted_rating))
        
        # Sort and format
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        final_recommendations = []
        for isbn, pred_rating in recommendations[:n_recommendations]:
            book_info = books_clean[books_clean['ISBN'] == isbn]
            if not book_info.empty:
                final_recommendations.append({
                    'isbn': isbn,
                    'title': book_info['Book-Title'].iloc[0],
                    'author': book_info['Book-Author'].iloc[0],
                    'year': int(book_info['Year-Of-Publication'].iloc[0]),
                    'publisher': book_info['Publisher'].iloc[0] if pd.notna(book_info['Publisher'].iloc[0]) else "Unknown",
                    'predicted_rating': float(pred_rating),
                    'image_url': book_info['Image-URL-M'].iloc[0] if pd.notna(book_info['Image-URL-M'].iloc[0]) else ""
                })
        
        return {"recommendations": final_recommendations}
        
    except Exception as e:
        return {"error": str(e)}

# ===================================================================
# API ENDPOINTS
# ===================================================================

@app.route('/')
def home():
    return jsonify({
        "message": "📚 Enhanced Book Recommendation API",
        "status": "running",
        "models_loaded": is_loaded,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0"
    })

@app.route('/status')
def status():
    """Get API status and model information"""
    if not is_loaded:
        return jsonify({"error": "Models not loaded"}), 500
    
    return jsonify({
        "status": "healthy",
        "models_loaded": True,
        "total_users": len(models['user_to_idx']),
        "total_books": len(models['book_to_idx']),
        "total_ratings": len(models['ratings_filtered']),
        "available_endpoints": [
            "/recommend/user/<user_id>",
            "/recommend/similar/<isbn>",
            "/recommend/svd/<user_id>",
            "/recommend/popular",
            "/recommend/genre/<genre>",
            "/book/<isbn>",
            "/search?q=<query>",
            "/user/<user_id>/ratings",
            "/genres"
        ]
    })

@app.route('/recommend/user/<int:user_id>')
def recommend_user(user_id):
    """Get personalized recommendations for a user"""
    n_recs = request.args.get('count', 10, type=int)
    n_recs = min(max(n_recs, 1), 50)  # Limit between 1 and 50
    
    result = get_user_recommendations_api(user_id, n_recs)
    
    if "error" in result:
        return jsonify(result), 404
    
    return jsonify({
        "user_id": user_id,
        "method": "collaborative_filtering",
        "count": len(result["recommendations"]),
        **result
    })

@app.route('/recommend/similar/<isbn>')
def recommend_similar(isbn):
    """Get books similar to a given book"""
    n_recs = request.args.get('count', 10, type=int)
    n_recs = min(max(n_recs, 1), 50)
    
    result = get_similar_books_api(isbn, n_recs)
    
    if "error" in result:
        return jsonify(result), 404
    
    return jsonify({
        "source_isbn": isbn,
        "method": "item_similarity",
        "count": len(result["recommendations"]),
        **result
    })

@app.route('/recommend/svd/<int:user_id>')
def recommend_svd(user_id):
    """Get SVD-based recommendations for a user"""
    n_recs = request.args.get('count', 10, type=int)
    n_recs = min(max(n_recs, 1), 50)
    
    result = get_svd_recommendations_api(user_id, n_recs)
    
    if "error" in result:
        return jsonify(result), 404
    
    return jsonify({
        "user_id": user_id,
        "method": "matrix_factorization",
        "count": len(result["recommendations"]),
        **result
    })

@app.route('/recommend/popular')
def recommend_popular():
    """Get popular books as fallback recommendations"""
    try:
        if not is_loaded:
            return jsonify({"error": "Models not loaded"}), 500
        
        n_recs = request.args.get('count', 10, type=int)
        n_recs = min(max(n_recs, 1), 50)
        
        popular_books = models['popular_books'][:n_recs]
        
        return jsonify({
            "method": "popularity_based",
            "count": len(popular_books),
            "recommendations": popular_books
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/recommend/genre/<genre>')
def recommend_by_genre(genre):
    """Get book recommendations by genre"""
    try:
        if not is_loaded:
            return jsonify({"error": "Models not loaded"}), 500
        
        books_clean = models['books_clean']
        ratings_filtered = models['ratings_filtered']
        
        # Search for books that contain the genre in title, author, or publisher
        genre_lower = genre.lower()
        genre_mask = (
            books_clean['Book-Title'].str.lower().str.contains(genre_lower, na=False) |
            books_clean['Book-Author'].str.lower().str.contains(genre_lower, na=False) |
            books_clean['Publisher'].str.lower().str.contains(genre_lower, na=False)
        )
        
        genre_books = books_clean[genre_mask]
        
        if genre_books.empty:
            return jsonify({"recommendations": []})
        
        # Get ratings for these books and sort by popularity and rating
        book_stats = []
        for _, book in genre_books.iterrows():
            book_ratings = ratings_filtered[ratings_filtered['ISBN'] == book['ISBN']]
            if len(book_ratings) >= 5:  # At least 5 ratings
                avg_rating = book_ratings['Book-Rating'].mean()
                rating_count = len(book_ratings)
                
                book_stats.append({
                    'isbn': book['ISBN'],
                    'title': book['Book-Title'],
                    'author': book['Book-Author'],
                    'year': int(book['Year-Of-Publication']),
                    'publisher': book['Publisher'] if pd.notna(book['Publisher']) else "Unknown",
                    'image_url': book['Image-URL-M'] if pd.notna(book['Image-URL-M']) else "",
                    'average_rating': float(avg_rating),
                    'rating_count': int(rating_count),
                    'popularity_score': float(avg_rating * np.log(rating_count + 1))
                })
        
        # Sort by popularity score (rating * log(count))
        book_stats.sort(key=lambda x: x['popularity_score'], reverse=True)
        
        return jsonify({
            "genre": genre,
            "count": len(book_stats[:20]),
            "recommendations": book_stats[:20]
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/genres')
def get_popular_genres():
    """Get list of popular genres/categories"""
    popular_genres = [
        "Fiction", "Mystery", "Romance", "Science Fiction", "Fantasy", 
        "Thriller", "Historical", "Biography", "Self-Help", "Business",
        "Psychology", "Philosophy", "History", "Art", "Cooking",
        "Travel", "Poetry", "Drama", "Horror", "Adventure"
    ]
    return jsonify({"genres": popular_genres})

@app.route('/book/<isbn>')
def get_book_details(isbn):
    """Get detailed information about a specific book"""
    try:
        if not is_loaded:
            return jsonify({"error": "Models not loaded"}), 500
        
        books_clean = models['books_clean']
        ratings_filtered = models['ratings_filtered']
        
        book = books_clean[books_clean['ISBN'] == isbn]
        if book.empty:
            return jsonify({"error": "Book not found"}), 404
        
        # Get rating statistics
        book_ratings = ratings_filtered[ratings_filtered['ISBN'] == isbn]
        
        book_data = {
            "isbn": isbn,
            "title": book['Book-Title'].iloc[0],
            "author": book['Book-Author'].iloc[0],
            "year": int(book['Year-Of-Publication'].iloc[0]),
            "publisher": book['Publisher'].iloc[0] if pd.notna(book['Publisher'].iloc[0]) else "Unknown",
            "image_url_small": book['Image-URL-S'].iloc[0] if pd.notna(book['Image-URL-S'].iloc[0]) else "",
            "image_url_medium": book['Image-URL-M'].iloc[0] if pd.notna(book['Image-URL-M'].iloc[0]) else "",
            "image_url_large": book['Image-URL-L'].iloc[0] if pd.notna(book['Image-URL-L'].iloc[0]) else "",
            "rating_count": int(len(book_ratings)),
            "average_rating": float(book_ratings['Book-Rating'].mean()) if len(book_ratings) > 0 else 0,
            "rating_distribution": book_ratings['Book-Rating'].value_counts().to_dict() if len(book_ratings) > 0 else {}
        }
        
        return jsonify(book_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search')
def search_books():
    """Search books by title, author, or ISBN"""
    try:
        if not is_loaded:
            return jsonify({"error": "Models not loaded"}), 500
        
        query = request.args.get('q', '').lower().strip()
        if not query:
            return jsonify({"error": "Please provide a search query"}), 400
        
        books_clean = models['books_clean']
        
        # Search in title, author, and ISBN
        mask = (books_clean['Book-Title'].str.lower().str.contains(query, na=False)) | \
               (books_clean['Book-Author'].str.lower().str.contains(query, na=False)) | \
               (books_clean['ISBN'].str.lower().str.contains(query, na=False))
        
        results = books_clean[mask].head(50)  # Limit to 50 results
        
        search_results = []
        for _, book in results.iterrows():
            search_results.append({
                'isbn': book['ISBN'],
                'title': book['Book-Title'],
                'author': book['Book-Author'],
                'year': int(book['Year-Of-Publication']),
                'publisher': book['Publisher'] if pd.notna(book['Publisher']) else "Unknown",
                'image_url': book['Image-URL-M'] if pd.notna(book['Image-URL-M']) else ""
            })
        
        return jsonify({
            "query": query,
            "count": len(search_results),
            "results": search_results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/user/<int:user_id>/ratings')
def get_user_ratings(user_id):
    """Get ratings given by a specific user"""
    try:
        if not is_loaded:
            return jsonify({"error": "Models not loaded"}), 500
        
        user_item_matrix = models['user_item_matrix']
        books_clean = models['books_clean']
        
        if user_id not in user_item_matrix.index:
            return jsonify({"error": f"User {user_id} not found"}), 404
        
        user_ratings = user_item_matrix.loc[user_id]
        rated_books = user_ratings[user_ratings > 0]
        
        ratings_list = []
        for isbn, rating in rated_books.items():
            book_info = books_clean[books_clean['ISBN'] == isbn]
            if not book_info.empty:
                ratings_list.append({
                    'isbn': isbn,
                    'title': book_info['Book-Title'].iloc[0],
                    'author': book_info['Book-Author'].iloc[0],
                    'rating': int(rating),
                    'year': int(book_info['Year-Of-Publication'].iloc[0]),
                    'publisher': book_info['Publisher'].iloc[0] if pd.notna(book_info['Publisher'].iloc[0]) else "Unknown",
                    'image_url': book_info['Image-URL-M'].iloc[0] if pd.notna(book_info['Image-URL-M'].iloc[0]) else ""
                })
        
        # Sort by rating (highest first)
        ratings_list.sort(key=lambda x: x['rating'], reverse=True)
        
        return jsonify({
            "user_id": user_id,
            "total_ratings": len(ratings_list),
            "average_rating": float(rated_books.mean()),
            "ratings": ratings_list
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/random-user')
def get_random_user():
    """Get a random user ID for testing"""
    try:
        if not is_loaded:
            return jsonify({"error": "Models not loaded"}), 500
        
        user_to_idx = models['user_to_idx']
        random_user = np.random.choice(list(user_to_idx.keys()))
        
        return jsonify({
            "user_id": int(random_user),
            "message": "Use this user ID to test recommendations"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 STARTING ENHANCED BOOK RECOMMENDATION API")
    print("="*60)
    print(f"📊 Models loaded: {is_loaded}")
    if is_loaded:
        print(f"👥 Users in system: {len(models.get('user_to_idx', {}))}")
        print(f"📚 Books in system: {len(models.get('book_to_idx', {}))}")
    print("🌐 Server starting on http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, port=5000, host='0.0.0.0')
