import joblib
import json
import os
from typing import Dict, List, Optional

class MusicPlayerIntegration:
    def __init__(self, model_path: str = 'music_recommendation_model.joblib'):
        """Initialize the music player integration."""
        if os.path.exists(model_path):
            self.model_data = joblib.load(model_path)
            self.model = self.model_data['model']
            self.scaler = self.model_data['scaler']
            self.label_encoder = self.model_data['label_encoder']
        else:
            print("[DEBUG] Model file not found, skipping model loading.")
            self.model_data = None
            self.model = None
            self.scaler = None
            self.label_encoder = None
        self.recommendations_cache = {}
        # In-memory tracking
        self.played_songs = []  # List[Dict]
        self.liked_songs = []   # List[Dict]
        self.disliked_songs = [] # List[Dict]
        
    def get_youtube_music_id(self, track_name: str, artist_name: str) -> Optional[str]:
        """Get YouTube Music video ID for a given track."""
        try:
            from ytmusicapi import YTMusic
            ytmusic = YTMusic()
            
            # Search for the track
            search_results = ytmusic.search(f"{track_name} {artist_name}", filter='songs', limit=1)
            if search_results:
                return search_results[0]['videoId']
            return None
        except Exception as e:
            print(f"Error getting YouTube Music ID: {e}")
            return None
    
    def get_next_song(self, current_track: str, current_artist: str) -> Dict:
        """Get the next recommended song."""
        # Create cache key
        cache_key = f"{current_track}-{current_artist}"
        
        # Check cache first
        if cache_key in self.recommendations_cache:
            rec = self.recommendations_cache[cache_key].pop(0)
            if not self.recommendations_cache[cache_key]:  # If empty, remove from cache
                del self.recommendations_cache[cache_key]
            return rec
        
        # If not in cache, generate new recommendations
        try:
            # Load the training data to find the song index
            import pandas as pd
            data = pd.read_csv('train.csv')
            song_idx = data[
                (data['Track Name'].str.lower() == current_track.lower()) &
                (data['Artist Name'].str.lower() == current_artist.lower())
            ].index[0]
            
            # Get recommendations
            distances, indices = self.model.kneighbors(
                self.scaler.transform(data.iloc[song_idx][self.features]).reshape(1, -1),
                n_neighbors=6  # Get 5 recommendations plus the current song
            )
            
            # Store recommendations in cache
            recommendations = []
            for idx, dist in zip(indices[0][1:], distances[0][1:]):  # Skip first (current song)
                rec = {
                    'track_name': data.iloc[idx]['Track Name'],
                    'artist_name': data.iloc[idx]['Artist Name'],
                    'genre': data.iloc[idx]['Class'],
                    'similarity_score': 1 - dist,
                    'video_id': self.get_youtube_music_id(
                        data.iloc[idx]['Track Name'],
                        data.iloc[idx]['Artist Name']
                    )
                }
                recommendations.append(rec)
            
            self.recommendations_cache[cache_key] = recommendations[1:]  # Store remaining recommendations
            return recommendations[0]
            
        except Exception as e:
            print(f"Error getting next song: {e}")
            # Fallback to basic recommendation
            with open('recommendations.json', 'r') as f:
                recs = json.load(f)
            genre = '0'  # Default to first genre
            return {
                'track_name': recs[genre][0].split(' - ')[0],
                'artist_name': recs[genre][0].split(' - ')[1],
                'genre': genre,
                'similarity_score': 1.0,
                'video_id': None
            }
    
    def update_recommendations(self, track_name: str, artist_name: str, genre: str, feedback: str) -> None:
        """Update recommendations based on user feedback (played, liked, disliked)."""
        song_str = f"{track_name} - {artist_name}"
        # Write to assets/recommendations.json instead of dataset/
        rec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../assets/recommendations.json'))
        print(f"[DEBUG] Writing recommendations to: {rec_path}")
        # Load or initialize recommendations
        if os.path.exists(rec_path):
            with open(rec_path, 'r', encoding='utf-8') as f:
                try:
                    recommendations = json.load(f)
                except Exception:
                    recommendations = {}
        else:
            recommendations = {}
        # Ensure genre key exists
        if genre not in recommendations:
            recommendations[genre] = []
        genre_list = recommendations[genre]
        # Remove song if present (for all feedback types)
        genre_list = [s for s in genre_list if s != song_str]
        # Feedback logic
        if feedback == 'played':
            print(f"[DEBUG] Played song: {song_str} (genre {genre})")
            genre_list.append(song_str)
            self.played_songs.append({'title': track_name, 'genre': genre})
        elif feedback == 'liked':
            print(f"[DEBUG] Liked song: {song_str} (genre {genre})")
            genre_list.insert(0, song_str)
            self.liked_songs.append({'title': track_name, 'genre': genre})
        elif feedback == 'disliked':
            print(f"[DEBUG] Disliked song: {song_str} (genre {genre})")
            self.disliked_songs.append({'title': track_name, 'genre': genre})
        # Keep max 10 songs per genre
        recommendations[genre] = genre_list[:10]
        # Save back
        with open(rec_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2, ensure_ascii=False)
        # Debug: print in-memory and file state
        print(f"[DEBUG] Played songs: {self.played_songs}")
        print(f"[DEBUG] Liked songs: {self.liked_songs}")
        print(f"[DEBUG] Disliked songs: {self.disliked_songs}")
        print(f"[DEBUG] Current recommendations for genre {genre}: {recommendations[genre]}")

    def play_song(self, track_name: str, artist_name: str, genre: str) -> None:
        """Call when a song is played."""
        self.update_recommendations(track_name, artist_name, genre, feedback='played')

    def like_song(self, track_name: str, artist_name: str, genre: str) -> None:
        """Call when a song is liked."""
        self.update_recommendations(track_name, artist_name, genre, feedback='liked')

    def dislike_song(self, track_name: str, artist_name: str, genre: str) -> None:
        """Call when a song is disliked."""
        self.update_recommendations(track_name, artist_name, genre, feedback='disliked')

    def get_advanced_recommendations(self, genre: str, n: int = 5) -> list:
        """Return up-to-date, session-aware recommendations for the user."""
        rec_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../assets/recommendations.json'))
        if not os.path.exists(rec_path):
            print("[DEBUG] recommendations.json not found.")
            return []
        with open(rec_path, 'r', encoding='utf-8') as f:
            recommendations = json.load(f)
        genre_recs = recommendations.get(genre, [])

        # Gather session history
        played = set(f'{s["title"]} - {genre}' for s in self.played_songs if s['genre'] == genre)
        disliked = set(f'{s["title"]} - {genre}' for s in self.disliked_songs if s['genre'] == genre)
        liked_titles = set(s['title'] for s in self.liked_songs if s['genre'] == genre)

        # Filter out played and disliked songs
        filtered = [song for song in genre_recs if song not in played and song not in disliked]

        # Boost liked songs to the top
        liked_boost = [song for song in filtered if any(lt in song for lt in liked_titles)]
        others = [song for song in filtered if song not in liked_boost]

        # Combine, prioritizing liked songs
        result = liked_boost + others

        # Add diversity: if not enough, pull from other genres
        if len(result) < n:
            for g, recs in recommendations.items():
                if g == genre:
                    continue
                for song in recs:
                    # Only add if not played/disliked in any genre
                    if song not in played and song not in disliked and song not in result:
                        result.append(song)
                        if len(result) >= n:
                            break
                if len(result) >= n:
                    break

        print(f"[DEBUG] Advanced recommendations for genre {genre} (n={n}): {result[:n]}")
        return result[:n]