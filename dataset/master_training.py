#!/usr/bin/env python3
"""
Master Training Script for Enhanced Music Recommendation System
This script runs the complete data expansion and training pipeline
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
        
        end_time = time.time()
        print(f"✅ {description} completed in {end_time - start_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error in {description}: {e}")
        return False

def check_requirements():
    """Check if required packages are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'scikit-learn':
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All requirements satisfied")
    return True

def main():
    """Main training pipeline"""
    print("Enhanced Music Recommendation System - Master Training")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements not met. Please install missing packages.")
        return False
    
    # Step 1: Create enhanced dataset
    if not run_script('enhanced_data_collection.py', 'Creating Enhanced Dataset'):
        print("❌ Failed to create enhanced dataset")
        return False
    
    # Step 2: Train enhanced model
    if not run_script('enhanced_training.py', 'Training Enhanced Model'):
        print("❌ Failed to train enhanced model")
        return False
    
    # Step 3: Train genre predictor (if not exists)
    if not os.path.exists('genre_predictor.joblib'):
        if not run_script('genre_predictor_train.py', 'Training Genre Predictor'):
            print("❌ Failed to train genre predictor")
            return False
    
    # Step 4: Generate final recommendations
    if not run_script('preprocess_and_recommend.py', 'Generating Final Recommendations'):
        print("❌ Failed to generate final recommendations")
        return False
    
    print("\nMaster Training Pipeline Completed Successfully!")
    print("=" * 60)
    
    # Display results
    print("\nTraining Results:")
    
    # Check enhanced dataset
    if os.path.exists('enhanced_train.csv'):
        import pandas as pd
        df = pd.read_csv('enhanced_train.csv')
        print(f"✅ Enhanced dataset: {len(df)} songs, {df.shape[1]} features")
    
    # Check user behavior data
    if os.path.exists('user_behavior.csv'):
        df = pd.read_csv('user_behavior.csv')
        print(f"✅ User behavior data: {len(df)} interactions")
    
    # Check models
    if os.path.exists('enhanced_music_model.joblib'):
        print("✅ Enhanced model: Trained and saved")
    
    if os.path.exists('genre_predictor.joblib'):
        print("✅ Genre predictor: Trained and saved")
    
    # Check recommendations
    if os.path.exists('enhanced_recommendations.json'):
        print("✅ Enhanced recommendations: Generated")
    
    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nYour enhanced music recommendation system is ready!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
