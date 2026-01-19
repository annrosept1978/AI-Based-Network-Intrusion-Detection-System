"""
Predict Network Traffic using Trained NIDS Model
This script demonstrates how to use the trained model to classify new network traffic samples.
"""

import numpy as np
import pandas as pd
import joblib

def load_model_and_scaler():
    """
    Load the trained model and scaler from disk
    
    Returns:
        model: Trained Random Forest classifier
        scaler: Fitted StandardScaler
    """
    print("Loading trained model and scaler...")
    try:
        model = joblib.load('nids_model.pkl')
        scaler = joblib.load('scaler.pkl')
        print("Model and scaler loaded successfully!")
        return model, scaler
    except FileNotFoundError as e:
        print(f"Error: Model files not found. Please train the model first by running 'python nids_train_test.py'")
        print(f"Missing file: {e.filename}")
        exit(1)
    except Exception as e:
        print(f"Error loading model files: {e}")
        print("The model files may be corrupted. Please retrain the model.")
        exit(1)

def create_sample_traffic():
    """
    Create sample network traffic for prediction
    
    Returns:
        DataFrame with sample traffic features
    """
    # Example 1: Normal-looking traffic
    normal_sample = {
        'packet_rate': 95,
        'byte_rate': 4800,
        'connection_duration': 12,
        'failed_logins': 0,
        'num_connections': 4,
        'error_rate': 0.02,
        'protocol_type': 0,  # TCP
        'service_type': 0    # HTTP
    }
    
    # Example 2: Suspicious traffic (potential attack)
    suspicious_sample = {
        'packet_rate': 550,
        'byte_rate': 22000,
        'connection_duration': 1,
        'failed_logins': 8,
        'num_connections': 65,
        'error_rate': 0.35,
        'protocol_type': 2,  # ICMP
        'service_type': 3    # Other
    }
    
    # Example 3: Another normal sample
    normal_sample_2 = {
        'packet_rate': 110,
        'byte_rate': 5200,
        'connection_duration': 8,
        'failed_logins': 0,
        'num_connections': 6,
        'error_rate': 0.01,
        'protocol_type': 1,  # UDP
        'service_type': 1    # SSH
    }
    
    # Combine samples into a DataFrame
    samples = pd.DataFrame([normal_sample, suspicious_sample, normal_sample_2])
    
    return samples

def predict_traffic(model, scaler, samples):
    """
    Predict whether traffic samples are normal or attacks
    
    Args:
        model: Trained classifier
        scaler: Fitted scaler
        samples: DataFrame with traffic features
        
    Returns:
        predictions: Array of predictions (0=Normal, 1=Attack)
        probabilities: Array of prediction probabilities
    """
    # Normalize the features using the same scaler used during training
    samples_scaled = scaler.transform(samples)
    
    # Make predictions
    predictions = model.predict(samples_scaled)
    
    # Get prediction probabilities
    probabilities = model.predict_proba(samples_scaled)
    
    return predictions, probabilities

def display_results(samples, predictions, probabilities):
    """
    Display prediction results in a readable format
    
    Args:
        samples: Original traffic samples
        predictions: Predicted labels
        probabilities: Prediction probabilities
    """
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    class_names = ['Normal', 'Attack']
    
    for i, (idx, sample) in enumerate(samples.iterrows()):
        print(f"\nSample {i+1}:")
        print("-" * 80)
        print(f"  Packet Rate: {sample['packet_rate']:.1f} packets/sec")
        print(f"  Byte Rate: {sample['byte_rate']:.1f} bytes/sec")
        print(f"  Connection Duration: {sample['connection_duration']:.1f} seconds")
        print(f"  Failed Logins: {sample['failed_logins']}")
        print(f"  Number of Connections: {sample['num_connections']}")
        print(f"  Error Rate: {sample['error_rate']:.2%}")
        
        prediction = class_names[predictions[i]]
        confidence = probabilities[i][predictions[i]] * 100
        
        print(f"\n  PREDICTION: {prediction} (Confidence: {confidence:.2f}%)")
        print(f"  Probability - Normal: {probabilities[i][0]:.2%}, Attack: {probabilities[i][1]:.2%}")

def main():
    """Main function to demonstrate model prediction"""
    print("="*80)
    print("NETWORK INTRUSION DETECTION SYSTEM - PREDICTION DEMO")
    print("="*80)
    
    # Step 1: Load the trained model and scaler
    model, scaler = load_model_and_scaler()
    
    # Step 2: Create sample traffic data
    print("\nCreating sample network traffic...")
    samples = create_sample_traffic()
    
    # Step 3: Make predictions
    print("Making predictions...")
    predictions, probabilities = predict_traffic(model, scaler, samples)
    
    # Step 4: Display results
    display_results(samples, predictions, probabilities)
    
    print("\n" + "="*80)
    print("PREDICTION COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    main()
