import pandas as pd
import json

# Load the training data
train = pd.read_csv('train.csv')

# Check the columns
print('Columns:', train.columns.tolist())

# For demonstration, recommend top 5 most popular songs per genre (Class)
recommendations = {}

# Group by genre (Class)
for genre, group in train.groupby('Class'):
    # Sort by Popularity (descending)
    top_songs = group.sort_values('Popularity', ascending=False).head(5)
    # Use Track Name and Artist Name as identifier
    song_list = [
        f"{row['Track Name']} - {row['Artist Name']}" for _, row in top_songs.iterrows()
    ]
    recommendations[genre] = song_list

# Save recommendations to JSON
with open('recommendations.json', 'w') as f:
    json.dump(recommendations, f, indent=2)

print('Recommendations saved to recommendations.json')