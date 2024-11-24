from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text ,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

import matplotlib.pyplot as plt

# שלב 1: טעינת הנתונים
iris = load_iris()
X, y = iris.data, iris.target


# Convert it to a pandas DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
print(iris_df.head())

species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
iris_df['species'] = iris_df['species'].map(species_mapping)

# Print the entire DataFrame as a table with headers
print(iris_df.to_string(index=False))


# שלב 2: חלוקת הנתונים לסט אימון וסט בדיקה
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# שלב 3: יצירת עץ החלטה
clf = DecisionTreeClassifier(max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# שלב 4: חיזוי ובדיקת ביצועי המודל
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"דיוק המודל: {accuracy:.2f}")

# שלב 5: הצגת העץ (מודפס כטקסט)
tree_rules = export_text(clf, feature_names=iris.feature_names)
print("\nחוקי עץ ההחלטה:\n")
print(tree_rules)



plt.figure(figsize=(10, 8))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()


hard_coded_data = [[0.5, 0.4, 0.5, 1]]
prediction = clf.predict(hard_coded_data)
predicted_class = iris_df['species'][prediction[0]]
print(f"\nPrediction for hard-coded data point {hard_coded_data}: {predicted_class}")
