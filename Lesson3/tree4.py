import xgboost as xgb
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=123)
eval_metrics = ['logloss', 'error', 'auc']

# Define XGBoost model
model = xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=0.3, learning_rate=0.1,
                          max_depth=5, alpha=10, n_estimators=200)

eval_set = [(X_train, y_train), (X_test, y_test)]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
# Define the parameters
param = {
    'max_depth': 5, 'eta': 0.1, 'objective': 'binary:logistic',
     'eval_metric': ['logloss', 'error', 'auc'],
    'alpha': 10, 'colsample_bytree': 0.3
}

num_round = 200
evallist = [(dtest, 'eval'), (dtrain, 'train')]

bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=10)


# Predict the labels of the test set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


X_test_subset = X_test[:10]
y_test_subset = y_test[:10]

# Predicting with the model
y_pred_subset = model.predict(X_test_subset)

for i in range(len(X_test_subset)):
    print(f"Instance: {i + 1}")
    print(f"Predicted: {y_pred_subset[i]}, Actual: {y_test_subset[i]}")
