import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
import json
from typing import Dict, List, Tuple
import joblib

class MusicRecommendationSystem:
    def __init__(self, data_path: str = 'train.csv'):
        """Initialize the recommendation system."""
        self.data_path = data_path
        self.data = None
        self.features = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess_data(self) -> None:
        """Load and preprocess the dataset."""
        print("Loading and preprocessing data...")
        
        # Load the dataset
        self.data = pd.read_csv(self.data_path)
        
        # Basic preprocessing
        self.data['Track Name'] = self.data['Track Name'].fillna('Unknown')
        self.data['Artist Name'] = self.data['Artist Name'].fillna('Unknown')
        self.data['Popularity'] = self.data['Popularity'].fillna(self.data['Popularity'].mean())
        
        # Create feature matrix
        feature_columns = [
            'Popularity', 'Danceability', 'Energy', 'Key', 'Loudness',
            'Mode', 'Speechiness', 'Acousticness', 'Instrumentalness',
            'Liveness', 'Valence', 'Tempo'
        ]
        
        # Scale the features
        self.features = self.scaler.fit_transform(self.data[feature_columns])
        
        # Encode genres
        self.data['Genre_Encoded'] = self.label_encoder.fit_transform(self.data['Class'])
        
        print("Data preprocessing completed.")
        print(f"Dataset shape: {self.data.shape}")
        print("\nFeature statistics:")
        print(pd.DataFrame(self.features, columns=feature_columns).describe())
        
    def train_model(self, n_neighbors: int = 5) -> None:
        """Train the recommendation model using K-Nearest Neighbors."""
        print("\nTraining model...")
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.model.fit(self.features)
        print("Model training completed.")
        
    def get_recommendations(self, song_idx: int, n_recommendations: int = 5) -> List[Dict]:
        """Get song recommendations based on a given song index."""
        distances, indices = self.model.kneighbors(
            self.features[song_idx].reshape(1, -1),
            n_neighbors=n_recommendations + 1
        )
        
        # Skip the first result as it's the input song itself
        recommendations = []
        for idx in indices[0][1:]:
            recommendations.append({
                'track_name': self.data.iloc[idx]['Track Name'],
                'artist_name': self.data.iloc[idx]['Artist Name'],
                'genre': self.data.iloc[idx]['Class'],
                'popularity': self.data.iloc[idx]['Popularity'],
                'similarity_score': 1 - distances[0][recommendations.length + 1]  # Convert distance to similarity
            })
            
        return recommendations
    
    def generate_genre_recommendations(self) -> Dict[str, List[str]]:
        """Generate recommendations per genre using the trained model."""
        recommendations = {}
        
        for genre in self.data['Class'].unique():
            genre_songs = self.data[self.data['Class'] == genre]
            if len(genre_songs) >= 5:
                # Get the most popular song in the genre
                popular_song_idx = genre_songs['Popularity'].idxmax()
                genre_recs = self.get_recommendations(popular_song_idx, 5)
                
                # Format recommendations
                recommendations[str(genre)] = [
                    f"{rec['track_name']} - {rec['artist_name']}"
                    for rec in genre_recs
                ]
            
        return recommendations
    
    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the model using various metrics."""
        print("\nEvaluating model...")
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            self.features,
            self.data['Genre_Encoded'],
            test_size=0.2,
            random_state=42
        )
        
        # Train a classifier for evaluation
        from sklearn.neighbors import KNeighborsClassifier
        knn_classifier = KNeighborsClassifier(n_neighbors=5)
        knn_classifier.fit(X_train, y_train)
        y_pred = knn_classifier.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        return metrics
    
    def save_recommendations(self, output_path: str = 'recommendations.json') -> None:
        """Save recommendations to a JSON file."""
        recommendations = self.generate_genre_recommendations()
        
        with open(output_path, 'w') as f:
            json.dump(recommendations, f, indent=2)
            
        print(f"\nRecommendations saved to {output_path}")
        
    def save_model(self, model_path: str = 'music_recommendation_model.joblib') -> None:
        """Save the trained model."""
        if self.model is not None:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder
            }, model_path)
            print(f"\nModel saved to {model_path}")
        else:
            print("Error: Model not trained yet.")

def main():
    # Initialize and train the recommendation system
    recommender = MusicRecommendationSystem()
    recommender.load_and_preprocess_data()
    recommender.train_model()
    
    # Evaluate the model
    metrics = recommender.evaluate_model()
    
    # Generate and save recommendations
    recommender.save_recommendations()
    
    # Save the model for later use
    recommender.save_model()

if __name__ == "__main__":
    main()