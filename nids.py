"""
AI-Based Network Intrusion Detection System (NIDS)
Uses Random Forest classifier to detect network intrusions

This script demonstrates a simple machine learning approach to classify
network traffic as either Normal or Attack based on various network features.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """
    Load the network traffic dataset from CSV file
    
    Args:
        filepath: Path to the CSV file containing network data
        
    Returns:
        DataFrame containing the loaded data
    """
    print("Loading dataset...")
    data = pd.read_csv(filepath)
    print(f"Dataset loaded successfully! Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    return data


def preprocess_data(data):
    """
    Preprocess the data by encoding categorical variables
    
    Args:
        data: Raw dataset DataFrame
        
    Returns:
        Tuple of (features, labels, label_encoder, feature_names)
    """
    print("\nPreprocessing data...")
    
    # Create a copy to avoid modifying original data
    df = data.copy()
    
    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Store feature names for later use
    feature_names = X.columns.tolist()
    
    # Encode categorical features (protocol_type, service, flag)
    categorical_columns = ['protocol_type', 'service', 'flag']
    
    for col in categorical_columns:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Encode the target labels (Normal/Attack)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Features shape: {X.shape}")
    print(f"Labels distribution:")
    print(pd.Series(y).value_counts())
    
    return X, y_encoded, label_encoder, feature_names


def train_model(X_train, y_train):
    """
    Train a Random Forest classifier on the training data
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained Random Forest model
    """
    print("\nTraining Random Forest classifier...")
    
    # Initialize Random Forest with sensible parameters
    # n_estimators: number of trees in the forest
    # random_state: for reproducibility
    rf_classifier = RandomForestClassifier(
        n_estimators=100,  # 100 decision trees
        random_state=42,   # for reproducible results
        max_depth=10,      # maximum depth of trees
        n_jobs=-1          # use all CPU cores
    )
    
    # Train the model
    rf_classifier.fit(X_train, y_train)
    print("Model trained successfully!")
    
    return rf_classifier


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the trained model on test data
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
        label_encoder: LabelEncoder used for labels
        
    Returns:
        Accuracy score
    """
    print("\nEvaluating model...")
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    
    # Display detailed classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=label_encoder.classes_
    ))
    
    # Display confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nInterpretation:")
    print(f"  True Negatives (Normal classified as Normal): {cm[0][0] if len(cm) > 0 else 0}")
    print(f"  False Positives (Normal classified as Attack): {cm[0][1] if len(cm) > 0 else 0}")
    if len(cm) > 1:
        print(f"  False Negatives (Attack classified as Normal): {cm[1][0]}")
        print(f"  True Positives (Attack classified as Attack): {cm[1][1]}")
    
    return accuracy


def display_feature_importance(model, feature_names, top_n=10):
    """
    Display the most important features for classification
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        top_n: Number of top features to display
    """
    print(f"\nTop {top_n} Most Important Features:")
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Display top N features
    print(feature_importance_df.head(top_n).to_string(index=False))


def main():
    """
    Main function to orchestrate the NIDS workflow
    """
    print("="*60)
    print("AI-Based Network Intrusion Detection System")
    print("="*60)
    
    # Step 1: Load the dataset
    data = load_data('network_data.csv')
    
    # Step 2: Preprocess the data
    X, y, label_encoder, feature_names = preprocess_data(data)
    
    # Step 3: Split data into training and testing sets
    # 70% training, 30% testing
    print("\nSplitting data into training and testing sets (70-30 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.3,      # 30% for testing
        random_state=42,    # for reproducibility
        stratify=y          # maintain class distribution
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Step 4: Train the Random Forest model
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test, label_encoder)
    
    # Step 6: Display feature importance
    display_feature_importance(model, feature_names)
    
    # Summary
    print("\n" + "="*60)
    print("NIDS Training and Evaluation Complete!")
    print(f"Final Model Accuracy: {accuracy * 100:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
