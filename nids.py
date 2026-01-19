import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'duration': [10, 20, 30, 40, 50],
    'protocol': [1, 1, 0, 0, 1],
    'service': [0, 1, 0, 1, 0],
    'label': [0, 0, 1, 1, 0]  # 0 = Normal, 1 = Attack
}

df = pd.DataFrame(data)

X = df.drop('label', axis=1)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
