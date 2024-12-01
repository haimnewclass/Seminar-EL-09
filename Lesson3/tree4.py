import xgboost as xgb
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
# Load dataset
data = load_breast_cancer()

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=123)

# Convert data into DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define parameters
param = {
    'max_depth': 5,
    'eta': 0.1,
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'error', 'auc'],
    'alpha': 10,
    'colsample_bytree': 0.3
}
num_round = 200
evallist = [(dtest, 'eval'), (dtrain, 'train')]

# Train the model
bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)

# Predict and evaluate
y_pred_proba = bst.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)  # Convert probabilities to binary labels
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# Predict the first 10 test records
dtest_subset = xgb.DMatrix(X_test[:10])
y_pred_subset_proba = bst.predict(dtest_subset)
y_pred_subset = (y_pred_subset_proba > 0.5).astype(int)

print("\nTesting First 10 Records:")
for i in range(10):
    print(f"Instance {i + 1}: Predicted: {y_pred_subset[i]}, Actual: {y_test[i]}")
