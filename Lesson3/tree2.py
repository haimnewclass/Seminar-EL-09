import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# טעינת נתוני Iris
iris = load_iris()
X = iris.data
y = iris.target

# חלוקת הנתונים לסט אימון ובדיקה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# יצירת מודל Random Forest
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# תחזיות
y_pred = clf.predict(X_test)

# הערכת המודל
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

feature_importances = clf.feature_importances_
for feature, importance in zip(iris.feature_names, feature_importances):
    print(f"{feature}: {importance:.2f}")


import matplotlib.pyplot as plt

accuracies = []
for n in range(1, 101):
    clf = RandomForestClassifier(n_estimators=n, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

plt.plot(range(1, 101), accuracies)
plt.xlabel("Number of Trees")
plt.ylabel("Accuracy")
plt.title("Effect of Number of Trees on Accuracy")
plt.show()

hard_coded_data = [[0.5, 0.4, 0.5, 1]]
prediction = clf.predict(hard_coded_data)

species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
predicted_class_index = prediction[0]
predicted_class_name = species_mapping[predicted_class_index]

print(f"\nPrediction for hard-coded data point {hard_coded_data}: {predicted_class_name}")


