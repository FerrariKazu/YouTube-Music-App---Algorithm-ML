import pandas as pd
import numpy as np
import requests
import json
import time
from datetime import datetime, timedelta
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataCollector:
    def __init__(self):
        self.base_data = None
        self.enhanced_data = None
        self.user_behavior_data = None
        
    def load_base_data(self, data_path='train.csv'):
        """Load the base training data"""
        print("Loading base training data...")
        self.base_data = pd.read_csv(data_path)
        print(f"Base dataset: {len(self.base_data)} songs")
        return self.base_data
    
    def create_derived_features(self, df):
        """Create additional derived features from existing audio features"""
        print("Creating derived features...")
        
        # Energy + Valence = Danceability score
        df['energy_valence_score'] = df['energy'] * df['valence']
        
        # Tempo categories
        df['tempo_category'] = pd.cut(df['tempo'], 
                                     bins=[0, 60, 90, 120, 150, 200, 300], 
                                     labels=['very_slow', 'slow', 'medium', 'fast', 'very_fast', 'extreme'])
        
        # Loudness categories
        df['loudness_category'] = pd.cut(df['loudness'], 
                                        bins=[-60, -20, -10, 0], 
                                        labels=['quiet', 'medium', 'loud'])
        
        # Acousticness + Instrumentalness = Acoustic score
        df['acoustic_score'] = df['acousticness'] * df['instrumentalness']
        
        # Speechiness + Liveness = Live performance score
        df['live_score'] = df['speechiness'] * df['liveness']
        
        # Energy + Danceability = Party score
        df['party_score'] = df['energy'] * df['danceability']
        
        # Valence + Danceability = Happy score
        df['happy_score'] = df['valence'] * df['danceability']
        
        # Tempo + Energy = Intensity score
        df['intensity_score'] = (df['tempo'] / 200) * df['energy']
        
        # Popularity + Energy = Mainstream score
        df['mainstream_score'] = (df['Popularity'] / 100) * df['energy']
        
        print(f"Added {8} derived features")
        return df
    
    def expand_genres(self, df):
        """Expand genre classifications with more specific sub-genres"""
        print("Expanding genre classifications...")
        
        # Create detailed genre mapping
        genre_mapping = {
            '0': ['Pop', 'Pop Rock', 'Dance Pop', 'Indie Pop', 'Synth Pop'],
            '1': ['Alternative', 'Indie', 'Alternative Rock', 'Indie Rock', 'Post-Rock'],
            '2': ['Blues', 'Delta Blues', 'Chicago Blues', 'Electric Blues', 'Blues Rock'],
            '3': ['Bollywood', 'Indian Pop', 'Bhangra', 'Fusion', 'Desi Pop'],
            '4': ['Country', 'Folk', 'Bluegrass', 'Americana', 'Country Rock'],
            '5': ['Hip-Hop', 'Rap', 'Trap', 'R&B', 'Urban'],
            '6': ['Alternative Rock', 'Indie Rock', 'Post-Rock', 'Grunge', 'Punk'],
            '7': ['Classical', 'Orchestral', 'Chamber Music', 'Symphony', 'Opera'],
            '8': ['Rock', 'Classic Rock', 'Hard Rock', 'Metal', 'Progressive Rock'],
            '9': ['Pop Rock', 'Soft Rock', 'Adult Contemporary', 'Power Pop'],
            '10': ['Pop', 'Contemporary Pop', 'Mainstream Pop', 'Radio Pop']
        }
        
        # Create sub-genre column
        def get_sub_genre(genre_list):
            if genre_list and isinstance(genre_list, list):
                return np.random.choice(genre_list)
            return 'Unknown'
        
        df['sub_genre'] = df['Class'].map(genre_mapping).apply(get_sub_genre)
        
        # Create genre confidence score (simulate how confident we are in genre assignment)
        df['genre_confidence'] = np.random.uniform(0.6, 1.0, len(df))
        
        print(f"Expanded to {len(genre_mapping)} main genres with sub-genres")
        return df
    
    def add_temporal_features(self, df):
        """Add time-based features"""
        print("Adding temporal features...")
        
        # Simulate release dates based on popularity and genre
        current_year = 2024
        df['release_year'] = np.random.randint(1950, 2024, len(df))
        
        # Adjust release years based on genre (older genres tend to be older)
        genre_year_adjustments = {
            '0': 0, '1': -5, '2': -20, '3': -10, '4': -15,
            '5': -5, '6': -10, '7': -30, '8': -25, '9': -5, '10': 0
        }
        
        for genre, adjustment in genre_year_adjustments.items():
            mask = df['Class'] == genre
            df.loc[mask, 'release_year'] += adjustment
        
        # Ensure years are within reasonable bounds
        df['release_year'] = np.clip(df['release_year'], 1950, 2024)
        
        # Extract temporal features
        df['release_month'] = np.random.randint(1, 13, len(df))
        df['release_decade'] = (df['release_year'] // 10) * 10
        
        # Calculate age of song
        df['song_age'] = current_year - df['release_year']
        
        # Categorize by era
        df['era'] = pd.cut(df['release_year'], 
                          bins=[1900, 1960, 1980, 2000, 2010, 2024],
                          labels=['Pre-60s', '60s-70s', '80s-90s', '2000s', '2010s+'])
        
        # Add seasonal features
        df['season'] = df['release_month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        print("Added temporal features: release_year, release_month, release_decade, song_age, era, season")
        return df
    
    def augment_audio_features(self, df, noise_factor=0.05, num_variations=2):
        """Add noise to audio features to create variations"""
        print(f"Augmenting audio features with {num_variations} variations per song...")
        
        audio_columns = ['danceability', 'energy', 'valence', 'acousticness', 
                        'instrumentalness', 'liveness', 'speechiness', 'tempo', 'loudness']
        
        augmented_data = []
        
        for _, row in df.iterrows():
            # Original row
            augmented_data.append(row)
            
            # Create variations
            for _ in range(num_variations):
                new_row = row.copy()
                for col in audio_columns:
                    if col in new_row and pd.notna(new_row[col]):
                        noise = np.random.normal(0, noise_factor)
                        if col == 'tempo':
                            new_row[col] = max(0, new_row[col] + noise * 10)
                        elif col == 'loudness':
                            new_row[col] = max(-60, min(0, new_row[col] + noise * 5))
                        else:
                            new_row[col] = max(0, min(1, new_row[col] + noise))
                augmented_data.append(new_row)
        
        augmented_df = pd.DataFrame(augmented_data)
        print(f"Augmented dataset: {len(augmented_df)} songs (from {len(df)})")
        return augmented_df
    
    def generate_user_behavior_data(self, df, num_users=1000):
        """Generate simulated user listening data"""
        print(f"Generating user behavior data for {num_users} users...")
        
        user_data = []
        
        for user_id in range(num_users):
            # Randomly select songs this user has listened to
            num_songs = np.random.randint(10, 100)
            user_songs = df.sample(n=min(num_songs, len(df)))
            
            for _, song in user_songs.iterrows():
                # Simulate realistic user behavior
                play_count = np.random.poisson(5) + 1  # Most songs played 1-10 times
                rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.3, 0.3, 0.1])
                last_played = datetime.now() - timedelta(days=np.random.randint(0, 365))
                
                user_data.append({
                    'user_id': user_id,
                    'track_name': song['Track Name'],
                    'artist_name': song['Artist Name'],
                    'genre': song['Class'],
                    'play_count': play_count,
                    'rating': rating,
                    'last_played': last_played,
                    'skip_rate': np.random.uniform(0, 0.3),  # 0-30% skip rate
                    'completion_rate': np.random.uniform(0.7, 1.0)  # 70-100% completion
                })
        
        user_df = pd.DataFrame(user_data)
        print(f"Generated {len(user_df)} user interactions")
        return user_df
    
    def add_social_features(self, df):
        """Add social and popularity features"""
        print("Adding social features...")
        
        # Simulate social media engagement
        df['social_score'] = np.random.uniform(0, 100, len(df))
        df['viral_potential'] = np.random.uniform(0, 1, len(df))
        df['trending_score'] = np.random.uniform(0, 100, len(df))
        
        # Add playlist features
        df['playlist_count'] = np.random.poisson(5, len(df))
        df['collaborative_playlist_count'] = np.random.poisson(2, len(df))
        
        # Add discovery features
        df['discovery_score'] = np.random.uniform(0, 1, len(df))
        df['recommendation_count'] = np.random.poisson(3, len(df))
        
        print("Added social features: social_score, viral_potential, trending_score, playlist_count, discovery_score")
        return df
    
    def create_enhanced_dataset(self):
        """Create the complete enhanced dataset"""
        print("Creating enhanced music recommendation dataset...")
        
        # Load base data
        self.base_data = self.load_base_data()
        
        # Create derived features
        enhanced_data = self.create_derived_features(self.base_data)
        
        # Expand genres
        enhanced_data = self.expand_genres(enhanced_data)
        
        # Add temporal features
        enhanced_data = self.add_temporal_features(enhanced_data)
        
        # Add social features
        enhanced_data = self.add_social_features(enhanced_data)
        
        # Augment data
        enhanced_data = self.augment_audio_features(enhanced_data)
        
        # Generate user behavior data
        self.user_behavior_data = self.generate_user_behavior_data(enhanced_data)
        
        self.enhanced_data = enhanced_data
        
        print(f"\nEnhanced dataset created!")
        print(f"Dataset size: {len(enhanced_data)} songs")
        print(f"Features: {enhanced_data.shape[1]} columns")
        print(f"Genres: {enhanced_data['Class'].nunique()} unique genres")
        print(f"User interactions: {len(self.user_behavior_data)}")
        
        return enhanced_data, self.user_behavior_data
    
    def save_enhanced_data(self):
        """Save the enhanced datasets"""
        print("Saving enhanced datasets...")
        
        # Save enhanced music data
        self.enhanced_data.to_csv('enhanced_train.csv', index=False)
        print("Enhanced music data saved to: enhanced_train.csv")
        
        # Save user behavior data
        self.user_behavior_data.to_csv('user_behavior.csv', index=False)
        print("User behavior data saved to: user_behavior.csv")
        
        # Save feature summary
        feature_summary = {
            'total_songs': len(self.enhanced_data),
            'total_features': self.enhanced_data.shape[1],
            'genres': self.enhanced_data['Class'].nunique(),
            'users': self.user_behavior_data['user_id'].nunique(),
            'interactions': len(self.user_behavior_data),
            'features': list(self.enhanced_data.columns)
        }
        
        with open('feature_summary.json', 'w') as f:
            json.dump(feature_summary, f, indent=2)
        print("Feature summary saved to: feature_summary.json")

def main():
    """Main function to create enhanced dataset"""
    collector = EnhancedDataCollector()
    
    # Create enhanced dataset
    enhanced_data, user_data = collector.create_enhanced_dataset()
    
    # Save datasets
    collector.save_enhanced_data()
    
    print("\nEnhanced dataset creation complete!")
    print("Files created:")
    print("   - enhanced_train.csv (enhanced music data)")
    print("   - user_behavior.csv (user interaction data)")
    print("   - feature_summary.json (dataset summary)")

if __name__ == "__main__":
    main()
