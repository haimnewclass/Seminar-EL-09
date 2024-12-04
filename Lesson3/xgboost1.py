from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# נתונים
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# מודל
model = XGBClassifier(
    n_estimators=100,  # Number of trees
    learning_rate=0.1,  # Learning rate
    max_depth=5,  # Maximum tree depth
    eval_metric='logloss'  # Evaluation metric
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# הערכה
print("Accuracy:", accuracy_score(y_test, y_pred))

test_record = [[5.1, 3.5, 1.4, 0.2]]
test_record_pred = model.predict(test_record)

# Output prediction
print(f"Prediction for the test record is: {test_record_pred}")


# 10 test records
test_records = X_test[:10]
test_records_pred = model.predict(test_records)

# Output prediction
for i in range(10):
    print(f"Test #{i + 1}")
    print(f"Predicted: {test_records_pred[i]}")
    print(f"Actual: {y_test[i]}")
    print("----------")