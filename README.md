# AI-Based Network Intrusion Detection System (NIDS)

A simple machine learning-based Network Intrusion Detection System that uses Random Forest classification to detect network intrusions and classify network traffic as either **Normal** or **Attack**.

## 📋 Project Description

This project demonstrates how artificial intelligence and machine learning can be applied to cybersecurity, specifically for detecting network intrusions. The system analyzes network traffic patterns and features to identify potentially malicious activities.

The implementation uses a **Random Forest classifier**, an ensemble learning method that creates multiple decision trees and combines their predictions for improved accuracy and robustness.

## 🛠️ Technologies Used

- **Python 3.7+**: Programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning library
  - Random Forest Classifier
  - Data preprocessing tools
  - Model evaluation metrics

## 📊 Dataset

The project includes a sample dataset (`network_data.csv`) containing network traffic features based on the KDD Cup 99 dataset structure. The dataset includes:

- **40 features** describing network connections:
  - Basic features: duration, protocol type, service, flag
  - Content features: logged_in, num_file_creations, num_shells, etc.
  - Traffic features: count, srv_count, error rates
  - Host-based features: dst_host_count, dst_host_srv_count, etc.

- **2 classes**:
  - `Normal`: Legitimate network traffic
  - `Attack`: Malicious network activity

The sample dataset contains 41 records for demonstration purposes.

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/annrosept1978/AI-Based-Network-Intrusion-Detection-System.git
   cd AI-Based-Network-Intrusion-Detection-System
   ```

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install packages individually:
   ```bash
   pip install pandas numpy scikit-learn
   ```

## 💻 How to Run

1. **Ensure you have the dataset**: The `network_data.csv` file should be in the same directory as `nids.py`

2. **Run the NIDS script**:
   ```bash
   python nids.py
   ```

3. **View the results**: The script will:
   - Load and preprocess the dataset
   - Split data into training (70%) and testing (30%) sets
   - Train a Random Forest classifier
   - Display accuracy metrics and classification report
   - Show feature importance rankings

## 📈 Expected Output

When you run the script, you'll see:

1. **Dataset Information**: Size, shape, and feature columns
2. **Preprocessing Details**: Feature encoding and label distribution
3. **Training Progress**: Model training confirmation
4. **Evaluation Metrics**:
   - Overall accuracy percentage
   - Classification report (precision, recall, F1-score)
   - Confusion matrix
5. **Feature Importance**: Top 10 features most important for classification

## 🎓 Learning Objectives

This project is designed for learning purposes and demonstrates:

- Loading and preprocessing network data
- Encoding categorical variables
- Splitting data for training and testing
- Training a Random Forest classifier
- Evaluating model performance with multiple metrics
- Understanding feature importance in ML models
- Applying ML to cybersecurity problems

## 🔍 Code Structure

- `nids.py`: Main script containing all functions
  - `load_data()`: Loads the CSV dataset
  - `preprocess_data()`: Encodes categorical features and labels
  - `train_model()`: Trains the Random Forest classifier
  - `evaluate_model()`: Calculates accuracy and displays metrics
  - `display_feature_importance()`: Shows most important features
  - `main()`: Orchestrates the entire workflow

- `network_data.csv`: Sample dataset with network traffic records
- `requirements.txt`: Python package dependencies

## 🔧 Customization

You can customize the model by modifying parameters in the `train_model()` function:

- `n_estimators`: Number of trees in the forest (default: 100)
- `max_depth`: Maximum depth of trees (default: 10)
- `random_state`: Seed for reproducibility (default: 42)

## 📝 Notes

- The sample dataset is intentionally small for demonstration purposes
- In production, you would use a much larger dataset for better accuracy
- The model achieves high accuracy on this sample dataset
- Consider adding more preprocessing steps and feature engineering for real-world applications

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements!

## 📄 License

This project is open source and available for educational purposes.

## 👨‍💻 Author

Created as a learning project to demonstrate AI-based network intrusion detection.
