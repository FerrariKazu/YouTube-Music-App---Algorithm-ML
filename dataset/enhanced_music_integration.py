import joblib
import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class EnhancedMusicPlayerIntegration:
    def __init__(self, model_path='enhanced_music_model.joblib'):
        """Initialize the enhanced music player integration."""
        self.model_path = model_path
        self.model_data = None
        self.recommendations_cache = {}
        
        # In-memory tracking
        self.played_songs = []
        self.liked_songs = []
        self.disliked_songs = []
        self.user_preferences = defaultdict(float)
        
        # Load model if available
        self.load_model()
        
    def load_model(self):
        """Load the enhanced model if available"""
        if os.path.exists(self.model_path):
            try:
                self.model_data = joblib.load(self.model_path)
                print("✅ Enhanced model loaded successfully")
            except Exception as e:
                print(f"⚠️  Error loading enhanced model: {e}")
                self.model_data = None
        else:
            print("⚠️  Enhanced model not found, using basic mode")
            self.model_data = None
    
    def get_enhanced_recommendations(self, track_name: str, artist_name: str, genre: str, n: int = 10) -> List[Dict]:
        """Get enhanced recommendations using the ensemble model"""
        if not self.model_data:
            return self.get_basic_recommendations(genre, n)
        
        try:
            # Find song in dataset
            song_idx = self.find_song_index(track_name, artist_name)
            if song_idx is None:
                return self.get_basic_recommendations(genre, n)
            
            # Get enhanced recommendations
            recommendations = self.get_ensemble_recommendations(song_idx, n)
            
            # Apply user preferences
            recommendations = self.apply_user_preferences(recommendations)
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting enhanced recommendations: {e}")
            return self.get_basic_recommendations(genre, n)
    
    def find_song_index(self, track_name: str, artist_name: str) -> Optional[int]:
        """Find song index in the dataset"""
        # This would need access to the original dataset
        # For now, return None to use basic recommendations
        return None
    
    def get_ensemble_recommendations(self, song_idx: int, n: int) -> List[Dict]:
        """Get recommendations using ensemble approach"""
        # This would use the loaded model data
        # For now, return basic recommendations
        return []
    
    def get_basic_recommendations(self, genre: str, n: int) -> List[Dict]:
        """Get basic recommendations from JSON file"""
        rec_path = os.path.join(os.path.dirname(__file__), '../assets/recommendations.json')
        if not os.path.exists(rec_path):
            return []
        
        with open(rec_path, 'r', encoding='utf-8') as f:
            recommendations = json.load(f)
        
        genre_recs = recommendations.get(genre, [])
        
        # Filter out played and disliked songs
        played = set(f'{s["title"]} - {s["genre"]}' for s in self.played_songs if s['genre'] == genre)
        disliked = set(f'{s["title"]} - {s["genre"]}' for s in self.disliked_songs if s['genre'] == genre)
        
        filtered = [song for song in genre_recs if song not in played and song not in disliked]
        
        # Boost liked songs
        liked_titles = set(s['title'] for s in self.liked_songs if s['genre'] == genre)
        liked_boost = [song for song in filtered if any(lt in song for lt in liked_titles)]
        others = [song for song in filtered if song not in liked_boost]
        
        result = liked_boost + others
        
        # Add diversity from other genres if needed
        if len(result) < n:
            for g, recs in recommendations.items():
                if g == genre:
                    continue
                for song in recs:
                    if song not in played and song not in disliked and song not in result:
                        result.append(song)
                        if len(result) >= n:
                            break
                if len(result) >= n:
                    break
        
        return [{'track_name': song.split(' - ')[0], 'artist_name': song.split(' - ')[1], 'genre': genre} 
                for song in result[:n]]
    
    def apply_user_preferences(self, recommendations: List[Dict]) -> List[Dict]:
        """Apply user preferences to recommendations"""
        for rec in recommendations:
            # Boost based on user preferences
            genre = rec.get('genre', '0')
            artist = rec.get('artist_name', '')
            
            # Apply genre preference
            genre_boost = self.user_preferences.get(f'genre_{genre}', 1.0)
            
            # Apply artist preference
            artist_boost = self.user_preferences.get(f'artist_{artist.lower()}', 1.0)
            
            # Calculate final score
            rec['preference_score'] = rec.get('similarity_score', 1.0) * genre_boost * artist_boost
        
        # Sort by preference score
        return sorted(recommendations, key=lambda x: x.get('preference_score', 0), reverse=True)
    
    def update_user_preferences(self, track_name: str, artist_name: str, genre: str, action: str):
        """Update user preferences based on actions"""
        if action == 'liked':
            # Boost genre and artist preferences
            self.user_preferences[f'genre_{genre}'] += 0.1
            self.user_preferences[f'artist_{artist_name.lower()}'] += 0.05
            
        elif action == 'disliked':
            # Reduce genre and artist preferences
            self.user_preferences[f'genre_{genre}'] = max(0.1, self.user_preferences[f'genre_{genre}'] - 0.05)
            self.user_preferences[f'artist_{artist_name.lower()}'] = max(0.1, self.user_preferences[f'artist_{artist_name.lower()}'] - 0.02)
    
    def update_recommendations(self, track_name: str, artist_name: str, genre: str, feedback: str) -> None:
        """Update recommendations based on user feedback with enhanced features"""
        song_str = f"{track_name} - {artist_name}"
        rec_path = os.path.join(os.path.dirname(__file__), '../assets/recommendations.json')
        
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
        
        # Remove song if present
        genre_list = [s for s in genre_list if s != song_str]
        
        # Enhanced feedback logic
        if feedback == 'played':
            print(f"[DEBUG] Played song: {song_str} (genre {genre})")
            genre_list.append(song_str)
            self.played_songs.append({'title': track_name, 'genre': genre, 'artist': artist_name})
            
        elif feedback == 'liked':
            print(f"[DEBUG] Liked song: {song_str} (genre {genre})")
            genre_list.insert(0, song_str)  # Move to top
            self.liked_songs.append({'title': track_name, 'genre': genre, 'artist': artist_name})
            self.update_user_preferences(track_name, artist_name, genre, 'liked')
            
        elif feedback == 'disliked':
            print(f"[DEBUG] Disliked song: {song_str} (genre {genre})")
            self.disliked_songs.append({'title': track_name, 'genre': genre, 'artist': artist_name})
            self.update_user_preferences(track_name, artist_name, genre, 'disliked')
        
        # Keep max 15 songs per genre (increased from 10)
        recommendations[genre] = genre_list[:15]
        
        # Save back
        with open(rec_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2, ensure_ascii=False)
        
        # Enhanced debug output
        print(f"[DEBUG] Played songs: {len(self.played_songs)}")
        print(f"[DEBUG] Liked songs: {len(self.liked_songs)}")
        print(f"[DEBUG] Disliked songs: {len(self.disliked_songs)}")
        print(f"[DEBUG] User preferences: {dict(self.user_preferences)}")
        print(f"[DEBUG] Current recommendations for genre {genre}: {recommendations[genre][:5]}")
    
    def get_user_insights(self) -> Dict:
        """Get insights about user's music preferences"""
        insights = {
            'total_played': len(self.played_songs),
            'total_liked': len(self.liked_songs),
            'total_disliked': len(self.disliked_songs),
            'favorite_genres': Counter(s['genre'] for s in self.liked_songs),
            'favorite_artists': Counter(s['artist'] for s in self.liked_songs),
            'genre_distribution': Counter(s['genre'] for s in self.played_songs),
            'preference_scores': dict(self.user_preferences)
        }
        
        return insights
    
    def play_song(self, track_name: str, artist_name: str, genre: str) -> None:
        """Call when a song is played"""
        self.update_recommendations(track_name, artist_name, genre, feedback='played')

    def like_song(self, track_name: str, artist_name: str, genre: str) -> None:
        """Call when a song is liked"""
        self.update_recommendations(track_name, artist_name, genre, feedback='liked')

    def dislike_song(self, track_name: str, artist_name: str, genre: str) -> None:
        """Call when a song is disliked"""
        self.update_recommendations(track_name, artist_name, genre, feedback='disliked')
    
    def get_personalized_recommendations(self, genre: str, n: int = 10) -> List[Dict]:
        """Get personalized recommendations based on user history"""
        return self.get_enhanced_recommendations('', '', genre, n)
