from flask import Flask, request, jsonify
from dataset.music_player_integration import MusicPlayerIntegration
import joblib
import os

app = Flask(__name__)
player = MusicPlayerIntegration()

# Load genre predictor model
GENRE_MODEL_PATH = os.path.join('dataset', 'genre_predictor.joblib')
if os.path.exists(GENRE_MODEL_PATH):
    try:
        genre_model = joblib.load(GENRE_MODEL_PATH)
        print(f"✅ Loaded genre predictor model with {len(genre_model['genre_counts'])} genres")
    except Exception as e:
        print(f"⚠️  Error loading genre model: {e}")
        genre_model = None
else:
    genre_model = None
    print("⚠️  Genre predictor model not found. Run: python dataset/genre_predictor_train.py")

def predict_genre(track, artist):
    if not genre_model:
        return '0'
    
    text = f'{track} {artist}'
    
    # Simple text preprocessing (same as in training)
    import re
    def preprocess_text(text):
        text = re.sub(r'[^a-z\s]', ' ', text.lower())
        words = [word for word in text.split() if len(word) > 2]
        return ' '.join(words)
    
    processed_text = preprocess_text(text)
    words = processed_text.split()
    
    best_genre = '0'
    best_score = -float('inf')
    
    for genre in genre_model['genre_counts']:
        score = 0
        for word in words:
            if word in genre_model['genre_word_probs'][genre]:
                score += genre_model['genre_word_probs'][genre][word]
        if score > best_score:
            best_score = score
            best_genre = genre
    
    return best_genre

@app.route('/play', methods=['POST'])
def play():
    data = request.json
    track = data['track']
    artist = data['artist']
    genre = data.get('genre', '0')
    # If genre is '0' or empty, predict it
    if not genre or genre == '0':
        genre = predict_genre(track, artist)
    player.play_song(track, artist, genre)
    return '', 204

@app.route('/like', methods=['POST'])
def like():
    data = request.json
    player.like_song(data['track'], data['artist'], data['genre'])
    return '', 204

@app.route('/dislike', methods=['POST'])
def dislike():
    data = request.json
    player.dislike_song(data['track'], data['artist'], data['genre'])
    return '', 204

@app.route('/recommendations', methods=['GET'])
def recommendations():
    genre = request.args.get('genre', '0')
    n = int(request.args.get('n', 5))
    recs = player.get_advanced_recommendations(genre, n)
    return jsonify({'recommendations': recs})

if __name__ == '__main__':
    app.run(port=5000)
