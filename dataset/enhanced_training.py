import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class EnhancedMusicRecommendationSystem:
    def __init__(self, data_path='enhanced_train.csv'):
        """Initialize the enhanced recommendation system."""
        self.data_path = data_path
        self.data = None
        self.features = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.user_behavior_data = None
        
    def load_enhanced_data(self):
        """Load the enhanced dataset and user behavior data"""
        print("Loading enhanced datasets...")
        
        # Load enhanced music data
        self.data = pd.read_csv(self.data_path)
        print(f"Enhanced music data: {len(self.data)} songs")
        
        # Load user behavior data
        try:
            self.user_behavior_data = pd.read_csv('dataset/user_behavior.csv')
            print(f"User behavior data: {len(self.user_behavior_data)} interactions")
        except FileNotFoundError:
            print("User behavior data not found, continuing without it")
            self.user_behavior_data = None
        
        return self.data
    
    def create_advanced_features(self):
        """Create advanced features for better recommendations"""
        print("Creating advanced features...")
        
        # Audio feature combinations
        self.data['energy_valence_ratio'] = self.data['energy'] / (self.data['valence'] + 0.001)
        self.data['danceability_energy_product'] = self.data['danceability'] * self.data['energy']
        self.data['acoustic_instrumental_ratio'] = self.data['acousticness'] / (self.data['instrumentalness'] + 0.001)
        
        # Temporal features
        self.data['is_recent'] = (self.data['song_age'] < 5).astype(int)
        self.data['is_classic'] = (self.data['song_age'] > 20).astype(int)
        self.data['is_trending'] = (self.data['trending_score'] > 70).astype(int)
        
        # Popularity features
        self.data['popularity_category'] = pd.cut(self.data['Popularity'], 
                                                bins=[0, 20, 40, 60, 80, 100], 
                                                labels=['low', 'medium-low', 'medium', 'high', 'very_high'])
        
        # Genre diversity features
        genre_popularity = self.data.groupby('Class')['Popularity'].mean()
        self.data['genre_popularity'] = self.data['Class'].map(genre_popularity)
        
        # User behavior features (if available)
        if self.user_behavior_data is not None:
            # Calculate song popularity based on user interactions
            song_stats = self.user_behavior_data.groupby(['track_name', 'artist_name']).agg({
                'play_count': 'sum',
                'rating': 'mean',
                'user_id': 'nunique'
            }).reset_index()
            
            song_stats.columns = ['Track Name', 'Artist Name', 'total_plays', 'avg_rating', 'unique_users']
            
            # Merge with main data
            self.data = self.data.merge(song_stats, on=['Track Name', 'Artist Name'], how='left')
            self.data['total_plays'] = self.data['total_plays'].fillna(0)
            self.data['avg_rating'] = self.data['avg_rating'].fillna(3.0)
            self.data['unique_users'] = self.data['unique_users'].fillna(0)
        
        print("Advanced features created")
        return self.data
    
    def prepare_features(self):
        """Prepare features for machine learning"""
        print("Preparing features for ML...")
        
        # Select feature columns
        feature_columns = [
            'Popularity', 'danceability', 'energy', 'key', 'loudness',
            'mode', 'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'energy_valence_score',
            'acoustic_score', 'live_score', 'party_score', 'happy_score',
            'intensity_score', 'mainstream_score', 'social_score',
            'viral_potential', 'trending_score', 'discovery_score'
        ]
        
        # Add user behavior features if available
        if 'total_plays' in self.data.columns:
            feature_columns.extend(['total_plays', 'avg_rating', 'unique_users'])
        
        # Add temporal features
        feature_columns.extend(['song_age', 'is_recent', 'is_classic', 'is_trending'])
        
        # Filter existing columns
        feature_columns = [col for col in feature_columns if col in self.data.columns]
        
        # Handle missing values
        for col in feature_columns:
            if self.data[col].dtype in ['object', 'category']:
                self.data[col] = self.data[col].fillna('Unknown')
            else:
                self.data[col] = self.data[col].fillna(self.data[col].mean())
        
        # Scale features
        self.features = self.scaler.fit_transform(self.data[feature_columns])
        
        # Encode genres
        self.data['Genre_Encoded'] = self.label_encoder.fit_transform(self.data['Class'])
        
        print(f"Prepared {len(feature_columns)} features for ML")
        print(f"Feature columns: {feature_columns}")
        
        return self.features, feature_columns
    
    def train_ensemble_model(self):
        """Train an ensemble of models for better recommendations"""
        print("Training ensemble model...")
        
        # K-Nearest Neighbors for similarity
        self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
        self.knn_model.fit(self.features)
        
        # Random Forest for genre classification
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(self.features, self.data['Genre_Encoded'])
        
        # SVM for high-dimensional similarity
        self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        self.svm_model.fit(self.features, self.data['Genre_Encoded'])
        
        # K-Means for clustering
        self.kmeans_model = KMeans(n_clusters=20, random_state=42)
        self.data['cluster'] = self.kmeans_model.fit_predict(self.features)
        
        print("Ensemble model training completed")
    
    def get_enhanced_recommendations(self, song_idx, n_recommendations=10):
        """Get enhanced recommendations using ensemble approach"""
        # Get KNN recommendations
        distances, indices = self.knn_model.kneighbors(
            self.features[song_idx].reshape(1, -1),
            n_neighbors=n_recommendations + 1
        )
        
        recommendations = []
        for i, (idx, dist) in enumerate(zip(indices[0][1:], distances[0][1:])):
            song = self.data.iloc[idx]
            
            # Calculate ensemble score
            knn_score = 1 - dist
            genre_prob = self.rf_model.predict_proba(self.features[idx].reshape(1, -1))[0]
            genre_score = max(genre_prob)
            
            # Cluster similarity
            cluster_score = 1 if song['cluster'] == self.data.iloc[song_idx]['cluster'] else 0.5
            
            # User behavior score (if available)
            behavior_score = 1.0
            if 'avg_rating' in song and pd.notna(song['avg_rating']):
                behavior_score = song['avg_rating'] / 5.0
            
            # Combined score
            ensemble_score = (knn_score * 0.4 + genre_score * 0.3 + 
                            cluster_score * 0.2 + behavior_score * 0.1)
            
            recommendations.append({
                'track_name': song['Track Name'],
                'artist_name': song['Artist Name'],
                'genre': song['Class'],
                'sub_genre': song.get('sub_genre', 'Unknown'),
                'popularity': song['popularity'],
                'similarity_score': ensemble_score,
                'knn_score': knn_score,
                'genre_score': genre_score,
                'cluster_score': cluster_score,
                'behavior_score': behavior_score
            })
        
        return sorted(recommendations, key=lambda x: x['similarity_score'], reverse=True)
    
    def generate_enhanced_recommendations(self):
        """Generate enhanced recommendations per genre"""
        print("Generating enhanced recommendations...")
        
        recommendations = {}
        
        for genre in self.data['Class'].unique():
            genre_songs = self.data[self.data['Class'] == genre]
            if len(genre_songs) >= 5:
                # Get the most popular song in the genre
                popular_song_idx = genre_songs['popularity'].idxmax()
                genre_recs = self.get_enhanced_recommendations(popular_song_idx, 10)
                
                # Format recommendations
                recommendations[str(genre)] = [
                    f"{rec['track_name']} - {rec['artist_name']}"
                    for rec in genre_recs[:5]  # Top 5 per genre
                ]
        
        return recommendations
    
    def evaluate_enhanced_model(self):
        """Evaluate the enhanced model"""
        print("Evaluating enhanced model...")
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            self.features,
            self.data['Genre_Encoded'],
            test_size=0.2,
            random_state=42
        )
        
        # Train and evaluate Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'precision': precision_score(y_test, rf_pred, average='weighted'),
            'recall': recall_score(y_test, rf_pred, average='weighted'),
            'f1': f1_score(y_test, rf_pred, average='weighted')
        }
        
        print("\nEnhanced Model Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        return metrics
    
    def save_enhanced_model(self):
        """Save the enhanced model and data"""
        print("Saving enhanced model...")
        
        model_data = {
            'knn_model': self.knn_model,
            'rf_model': self.rf_model,
            'svm_model': self.svm_model,
            'kmeans_model': self.kmeans_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.features.shape[1]
        }
        
        joblib.dump(model_data, 'enhanced_music_model.joblib')
        print("Enhanced model saved to: enhanced_music_model.joblib")
        
        # Save enhanced recommendations
        recommendations = self.generate_enhanced_recommendations()
        with open('enhanced_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        print("Enhanced recommendations saved to: enhanced_recommendations.json")

def main():
    """Main function to train enhanced model"""
    print("Training Enhanced Music Recommendation System...")
    
    # Initialize system
    system = EnhancedMusicRecommendationSystem()
    
    # Load data
    system.load_enhanced_data()
    
    # Create advanced features
    system.create_advanced_features()
    
    # Prepare features
    system.prepare_features()
    
    # Train ensemble model
    system.train_ensemble_model()
    
    # Evaluate model
    metrics = system.evaluate_enhanced_model()
    
    # Save model and recommendations
    system.save_enhanced_model()
    
    print("\nEnhanced training complete!")
    print("Files created:")
    print("   - enhanced_music_model.joblib (trained model)")
    print("   - enhanced_recommendations.json (recommendations)")

if __name__ == "__main__":
    main()
