import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Step 1: Create a synthetic dataset
X, y = make_classification(
    n_samples=1000,   # Number of samples
    n_features=20,    # Number of features
    n_informative=15, # Number of informative features
    n_redundant=5,    # Number of redundant features
    random_state=42   # Random seed
)

# Step 2: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize the XGBoost classifier
model = XGBClassifier(
    n_estimators=100,     # Number of trees
    learning_rate=0.1,    # Learning rate
    max_depth=5,          # Maximum tree depth
    eval_metric='logloss' # Evaluation metric
)

# Step 4: Train the model
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Step 7: Test a single sample
single_sample = X_test[0].reshape(1, -1)  # Reshape for a single sample
single_prediction = model.predict(single_sample)
print(f"Single Sample Prediction: {single_prediction[0]}")
print(f"True Label: {y_test[0]}")