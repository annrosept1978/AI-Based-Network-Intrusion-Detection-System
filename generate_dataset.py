"""
Generate Sample Network Traffic Dataset
This script creates a synthetic dataset simulating network traffic features
for training the intrusion detection system.
"""

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

def generate_normal_traffic(n_samples):
    """
    Generate normal network traffic data
    
    Args:
        n_samples: Number of normal traffic samples to generate
        
    Returns:
        DataFrame with normal traffic features
    """
    # Normal traffic characteristics: Low packet rate, typical port numbers, low error rate
    data = {
        'packet_rate': np.random.normal(100, 20, n_samples),  # Average packet rate
        'byte_rate': np.random.normal(5000, 1000, n_samples),  # Average byte rate
        'connection_duration': np.random.exponential(10, n_samples),  # Connection duration in seconds
        'failed_logins': np.random.poisson(0.1, n_samples),  # Very few failed logins
        'num_connections': np.random.poisson(5, n_samples),  # Normal number of connections
        'error_rate': np.random.uniform(0, 0.05, n_samples),  # Low error rate
        'protocol_type': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]),  # TCP, UDP, ICMP
        'service_type': np.random.choice([0, 1, 2, 3], n_samples, p=[0.4, 0.3, 0.2, 0.1]),  # HTTP, SSH, FTP, Other
        'label': 0  # 0 = Normal
    }
    return pd.DataFrame(data)

def generate_attack_traffic(n_samples):
    """
    Generate attack network traffic data (DDoS, Port Scanning, Brute Force, etc.)
    
    Args:
        n_samples: Number of attack traffic samples to generate
        
    Returns:
        DataFrame with attack traffic features
    """
    # Attack traffic characteristics: High packet rate, unusual patterns, high error rate
    data = {
        'packet_rate': np.random.normal(500, 150, n_samples),  # Much higher packet rate
        'byte_rate': np.random.normal(20000, 8000, n_samples),  # Much higher byte rate
        'connection_duration': np.random.exponential(2, n_samples),  # Shorter connections
        'failed_logins': np.random.poisson(5, n_samples),  # Many failed login attempts
        'num_connections': np.random.poisson(50, n_samples),  # Unusually high number of connections
        'error_rate': np.random.uniform(0.1, 0.5, n_samples),  # High error rate
        'protocol_type': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.3, 0.4]),  # Different distribution
        'service_type': np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.2, 0.3, 0.3]),  # Different distribution
        'label': 1  # 1 = Attack
    }
    return pd.DataFrame(data)

def main():
    """Main function to generate and save the dataset"""
    print("Generating sample network traffic dataset...")
    
    # Generate 800 normal and 200 attack samples (imbalanced as in real-world scenarios)
    normal_data = generate_normal_traffic(800)
    attack_data = generate_attack_traffic(200)
    
    # Combine and shuffle the data
    dataset = pd.concat([normal_data, attack_data], ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to CSV file
    dataset.to_csv('network_traffic_data.csv', index=False)
    
    print(f"Dataset generated successfully!")
    print(f"Total samples: {len(dataset)}")
    print(f"Normal traffic samples: {len(normal_data)}")
    print(f"Attack traffic samples: {len(attack_data)}")
    print(f"Dataset saved to 'network_traffic_data.csv'")
    
    # Display basic statistics
    print("\nDataset Statistics:")
    print(dataset.describe())
    print("\nClass Distribution:")
    print(dataset['label'].value_counts())

if __name__ == "__main__":
    main()
