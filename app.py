from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# For now, we'll load data directly (you'll optimize this later)
try:
    # Adjust paths based on your directory structure
    books = pd.read_csv('../data/raw/Books.csv', encoding='latin-1', low_memory=False)
    ratings = pd.read_csv('../data/raw/Ratings.csv', encoding='latin-1', low_memory=False)
    users = pd.read_csv('../data/raw/Users.csv', encoding='latin-1', low_memory=False)
    
    print("Data loaded successfully!")
    print(f"Books: {len(books)} rows")
    print(f"Ratings: {len(ratings)} rows") 
    print(f"Users: {len(users)} rows")
    
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Make sure your CSV files are in the data/raw/ directory")
    books = pd.DataFrame()
    ratings = pd.DataFrame()
    users = pd.DataFrame()

@app.route('/')
def home():
    return jsonify({
        "message": "Book Recommendation API is running!",
        "status": "success",
        "data_loaded": not books.empty
    })

@app.route('/stats')
def get_stats():
    """Get basic statistics about the dataset"""
    if books.empty:
        return jsonify({"error": "Data not loaded"}), 500
    
    stats = {
        "total_books": len(books),
        "total_users": len(users),
        "total_ratings": len(ratings),
        "unique_users_with_ratings": ratings['User-ID'].nunique() if not ratings.empty else 0,
        "unique_books_with_ratings": ratings['ISBN'].nunique() if not ratings.empty else 0
    }
    
    return jsonify(stats)

@app.route('/books')
def get_books():
    """Get a sample of books"""
    if books.empty:
        return jsonify({"error": "Data not loaded"}), 500
    
    # Get first 10 books
    sample_books = books.head(10)
    books_list = []
    
    for _, book in sample_books.iterrows():
        books_list.append({
            "isbn": book.get('ISBN', ''),
            "title": book.get('Book-Title', ''),
            "author": book.get('Book-Author', ''),
            "year": book.get('Year-Of-Publication', ''),
            "publisher": book.get('Publisher', '')
        })
    
    return jsonify({"books": books_list})

@app.route('/book/<isbn>')
def get_book(isbn):
    """Get details of a specific book"""
    if books.empty:
        return jsonify({"error": "Data not loaded"}), 500
    
    try:
        book = books[books['ISBN'] == isbn]
        if book.empty:
            return jsonify({"error": "Book not found"}), 404
        
        book_data = {
            "isbn": isbn,
            "title": book['Book-Title'].iloc[0],
            "author": book['Book-Author'].iloc[0],
            "year": book['Year-Of-Publication'].iloc[0],
            "publisher": book['Publisher'].iloc[0]
        }
        
        return jsonify(book_data)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search')
def search_books():
    """Search books by title or author"""
    if books.empty:
        return jsonify({"error": "Data not loaded"}), 500
    
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify({"error": "Please provide a search query"}), 400
    
    # Search in title and author
    mask = (books['Book-Title'].str.lower().str.contains(query, na=False)) | \
           (books['Book-Author'].str.lower().str.contains(query, na=False))
    
    results = books[mask].head(20)  # Limit to 20 results
    
    search_results = []
    for _, book in results.iterrows():
        search_results.append({
            "isbn": book.get('ISBN', ''),
            "title": book.get('Book-Title', ''),
            "author": book.get('Book-Author', ''),
            "year": book.get('Year-Of-Publication', ''),
            "publisher": book.get('Publisher', '')
        })
    
    return jsonify({
        "query": query,
        "results": search_results,
        "total_found": len(search_results)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
