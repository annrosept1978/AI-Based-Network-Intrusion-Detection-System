# AI-Based Network Intrusion Detection System (NIDS)

## Overview
This project implements a simple yet effective **Network Intrusion Detection System (NIDS)** using machine learning. The system uses a **Random Forest classifier** to distinguish between normal network traffic and potential attacks.

The project is designed for **educational purposes** to help understand:
- How machine learning can be applied to cybersecurity
- The fundamentals of intrusion detection systems
- Random Forest classification algorithm
- Model training, testing, and evaluation

## Features
- **Synthetic Dataset Generation**: Creates realistic network traffic data with normal and attack patterns
- **Random Forest Classifier**: Robust ensemble learning algorithm for binary classification
- **Feature Engineering**: Uses 8 key network traffic features
- **Model Evaluation**: Comprehensive accuracy metrics, confusion matrix, and classification report
- **Well-Commented Code**: Detailed comments throughout for learning purposes
- **Model Persistence**: Save and load trained models for future use

## Technologies Used
- **Python 3.x**: Primary programming language
- **NumPy**: Numerical computing and array operations
- **Pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning library for Random Forest, preprocessing, and evaluation
- **joblib**: Model serialization and persistence

## Dataset Features
The system analyzes the following network traffic features:

1. **packet_rate**: Number of packets per second
2. **byte_rate**: Number of bytes per second
3. **connection_duration**: How long the connection lasts (in seconds)
4. **failed_logins**: Number of failed login attempts
5. **num_connections**: Number of simultaneous connections
6. **error_rate**: Percentage of errors in transmission
7. **protocol_type**: Protocol used (TCP, UDP, ICMP)
8. **service_type**: Service being accessed (HTTP, SSH, FTP, Other)

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/annrosept1978/AI-Based-Network-Intrusion-Detection-System.git
   cd AI-Based-Network-Intrusion-Detection-System
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Step 1: Generate the Dataset
First, generate the synthetic network traffic dataset:

```bash
python generate_dataset.py
```

This will create a file named `network_traffic_data.csv` containing 1000 samples (800 normal, 200 attack).

### Step 2: Train and Test the Model
Run the training and testing script:

```bash
python nids_train_test.py
```

This script will:
1. Load the dataset
2. Split it into training (80%) and testing (20%) sets
3. Normalize the features
4. Train a Random Forest classifier
5. Evaluate the model and display accuracy metrics
6. Save the trained model and scaler for future use

### Expected Output
You should see output similar to:
```
============================================================
NETWORK INTRUSION DETECTION SYSTEM (NIDS)
Using Random Forest Classifier
============================================================
Loading dataset from network_traffic_data.csv...
Dataset loaded successfully! Shape: (1000, 9)
...
Accuracy: 95.00%
```

## Project Structure
```
AI-Based-Network-Intrusion-Detection-System/
│
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── generate_dataset.py        # Script to generate synthetic dataset
├── nids_train_test.py        # Main training and testing script
├── network_traffic_data.csv  # Generated dataset (after running generate_dataset.py)
├── nids_model.pkl            # Trained model (after running nids_train_test.py)
└── scaler.pkl                # Feature scaler (after running nids_train_test.py)
```

## How It Works

### 1. Data Generation
The `generate_dataset.py` script creates synthetic network traffic with distinct patterns:
- **Normal Traffic**: Lower packet rates, typical error rates, standard connection patterns
- **Attack Traffic**: Higher packet rates, elevated error rates, suspicious connection patterns (e.g., DDoS, port scanning)

### 2. Model Training
The Random Forest classifier is an ensemble method that:
- Creates multiple decision trees
- Each tree votes on the classification
- Final prediction is based on majority vote
- Resistant to overfitting and handles non-linear patterns well

### 3. Evaluation Metrics
The system provides:
- **Accuracy**: Overall correctness of predictions
- **Confusion Matrix**: True positives, true negatives, false positives, false negatives
- **Precision**: How many predicted attacks were actual attacks
- **Recall**: How many actual attacks were detected
- **F1-Score**: Harmonic mean of precision and recall

## Learning Outcomes
By studying this project, you will learn:
- How to generate synthetic datasets for machine learning
- Data preprocessing techniques (normalization, train-test split)
- Implementation of Random Forest classifiers
- Model evaluation and performance metrics
- Practical application of ML in cybersecurity

## Future Enhancements
Potential improvements for this project:
- Use real-world datasets (e.g., KDD Cup 99, NSL-KDD, CICIDS2017)
- Implement multi-class classification (different attack types)
- Add feature importance analysis
- Implement real-time detection capabilities
- Create a web interface for visualization
- Compare with other algorithms (SVM, Neural Networks, XGBoost)

## License
This project is open-source and available for educational purposes.

## Contributing
Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## Acknowledgments
This project is created for educational purposes to demonstrate the application of machine learning in network security.

## Contact
For questions or feedback, please open an issue on the GitHub repository.
