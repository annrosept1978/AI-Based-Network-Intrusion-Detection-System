"""
Network Intrusion Detection System (NIDS) using Random Forest
This script trains a Random Forest classifier to detect network intrusions
and evaluates its performance on test data.
"""

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

# Set random seed for reproducibility
np.random.seed(42)

def load_data(filename='network_traffic_data.csv'):
    """
    Load the network traffic dataset from CSV file
    
    Args:
        filename: Path to the CSV file containing network traffic data
        
    Returns:
        DataFrame with the loaded data
    """
    print(f"Loading dataset from {filename}...")
    try:
        data = pd.read_csv(filename)
        print(f"Dataset loaded successfully! Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: Dataset file '{filename}' not found.")
        print("Please generate the dataset first by running 'python generate_dataset.py'")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: Dataset file '{filename}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

def preprocess_data(data):
    """
    Preprocess the data by separating features and labels
    
    Args:
        data: DataFrame containing the network traffic data
        
    Returns:
        X: Feature matrix
        y: Label vector
    """
    print("\nPreprocessing data...")
    
    # Separate features (X) and labels (y)
    X = data.drop('label', axis=1)
    y = data['label']
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    return X, y

def split_data(X, y, test_size=0.2):
    """
    Split the data into training and testing sets
    
    Args:
        X: Feature matrix
        y: Label vector
        test_size: Proportion of data to use for testing (default: 0.2)
        
    Returns:
        X_train, X_test, y_train, y_test: Training and testing sets
    """
    print(f"\nSplitting data into train ({int((1-test_size)*100)}%) and test ({int(test_size*100)}%) sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def normalize_data(X_train, X_test):
    """
    Normalize the features using StandardScaler
    
    Args:
        X_train: Training feature matrix
        X_test: Testing feature matrix
        
    Returns:
        X_train_scaled, X_test_scaled: Normalized feature matrices
        scaler: The fitted scaler object
    """
    print("\nNormalizing features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features normalized successfully!")
    
    return X_train_scaled, X_test_scaled, scaler

def train_model(X_train, y_train):
    """
    Train a Random Forest classifier
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        
    Returns:
        model: Trained Random Forest classifier
    """
    print("\nTraining Random Forest classifier...")
    
    # Create Random Forest classifier
    # n_estimators: Number of trees in the forest
    # max_depth: Maximum depth of each tree (prevents overfitting)
    #            Set to 10 to prevent overfitting on this small synthetic dataset
    # random_state: Ensures reproducibility
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("Model training completed!")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on test data
    
    Args:
        model: Trained classifier
        X_test: Testing feature matrix
        y_test: Testing labels
        
    Returns:
        accuracy: Model accuracy on test set
    """
    print("\n" + "="*60)
    print("EVALUATING MODEL PERFORMANCE")
    print("="*60)
    
    # Make predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy*100:.2f}%")
    
    # Display confusion matrix
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("               Normal  Attack")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Actual Normal    {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"Actual Attack    {cm[1][0]:4d}    {cm[1][1]:4d}")
    
    # Display detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))
    
    return accuracy

def save_model(model, scaler, model_filename='nids_model.pkl', scaler_filename='scaler.pkl'):
    """
    Save the trained model and scaler to disk
    
    Args:
        model: Trained classifier
        scaler: Fitted scaler
        model_filename: Name of file to save model
        scaler_filename: Name of file to save scaler
    """
    print("\nSaving model and scaler...")
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)
    print(f"Model saved to '{model_filename}'")
    print(f"Scaler saved to '{scaler_filename}'")

def main():
    """Main function to orchestrate the training and testing process"""
    print("="*60)
    print("NETWORK INTRUSION DETECTION SYSTEM (NIDS)")
    print("Using Random Forest Classifier")
    print("="*60)
    
    # Step 1: Load the dataset
    data = load_data('network_traffic_data.csv')
    
    # Step 2: Preprocess the data
    X, y = preprocess_data(data)
    
    # Step 3: Split into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Step 4: Normalize the features
    X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)
    
    # Step 5: Train the Random Forest model
    model = train_model(X_train_scaled, y_train)
    
    # Step 6: Evaluate the model
    accuracy = evaluate_model(model, X_test_scaled, y_test)
    
    # Step 7: Save the trained model and scaler
    save_model(model, scaler)
    
    print("\n" + "="*60)
    print("TRAINING AND TESTING COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()
