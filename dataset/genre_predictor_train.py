import pandas as pd
import joblib
import re
from collections import Counter, defaultdict

print("Loading training data...")
# Load training data
train_path = 'dataset/train.csv'
df = pd.read_csv(train_path)
print(f"Loaded {len(df)} songs from {train_path}")

print("Preparing text features...")
df['text'] = df['Track Name'].fillna('') + ' ' + df['Artist Name'].fillna('')
df['text'] = df['text'].str.lower()

# Simple text preprocessing
def preprocess_text(text):
    # Remove special characters, keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Split into words and remove empty strings
    words = [word for word in text.split() if len(word) > 2]
    return ' '.join(words)

df['processed_text'] = df['text'].apply(preprocess_text)

print("Building simple genre classifier...")
# Build a simple word-based genre classifier
genre_word_counts = defaultdict(lambda: defaultdict(int))
genre_counts = Counter()

for _, row in df.iterrows():
    genre = str(row['Class'])
    text = row['processed_text']
    words = text.split()
    genre_counts[genre] += 1
    
    for word in words:
        genre_word_counts[genre][word] += 1

# Calculate word probabilities for each genre
genre_word_probs = {}
total_words_per_genre = {}

for genre in genre_counts:
    total_words = sum(genre_word_counts[genre].values())
    total_words_per_genre[genre] = total_words
    genre_word_probs[genre] = {}
    
    for word, count in genre_word_counts[genre].items():
        # Add smoothing to avoid zero probabilities
        genre_word_probs[genre][word] = (count + 1) / (total_words + len(genre_word_counts[genre]))

def predict_genre_simple(text):
    text = preprocess_text(text.lower())
    words = text.split()
    
    best_genre = '0'
    best_score = -float('inf')
    
    for genre in genre_counts:
        score = 0
        for word in words:
            if word in genre_word_probs[genre]:
                score += genre_word_probs[genre][word]
        if score > best_score:
            best_score = score
            best_genre = genre
    
    return best_genre

print("Testing classifier...")
# Test accuracy
correct = 0
total = 0
test_df = df.head(100)  # Get first 100 songs
for _, row in test_df.iterrows():
    predicted = predict_genre_simple(row['text'])
    actual = str(row['Class'])
    if predicted == actual:
        correct += 1
    total += 1

print(f"Test accuracy on 100 songs: {correct/total:.2%}")

print("Saving model...")
# Save the simple model (without the function reference)
model_data = {
    'genre_word_probs': dict(genre_word_probs),
    'genre_counts': dict(genre_counts),
    'total_words_per_genre': total_words_per_genre
}

joblib.dump(model_data, 'dataset/genre_predictor.joblib')
print('✅ Simple genre predictor model saved to dataset/genre_predictor.joblib')
print(f"Model can predict {len(genre_counts)} different genres: {sorted(genre_counts.keys())}")
