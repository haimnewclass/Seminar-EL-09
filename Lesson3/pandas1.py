import pandas as pd
from sklearn.datasets import load_iris
import numpy as np

data = load_iris()

X, y = data.data, data.target

import pandas as pd

#DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'San Francisco', 'Los Angeles']
}
#from Json import
df = pd.DataFrame(data)
print(df)


array = np.array([[1, 2], [3, 4], [5, 6]])
df_from_array = pd.DataFrame(array, columns=['Name', 'Idx'])
print(df_from_array)

#from sklearn

iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['target'] = iris.target
print(df_iris.head())

print(df.head())  # חמש שורות ראשונות
print(df.tail())  # חמש שורות אחרונות
print(df.info())  # מידע כללי על הנתונים
print(df.describe())  # סטטיסטיקות בסיסיות לעמודות מספריות


print(df['Name'])  # מחזיר את עמודת Name


print(df.iloc[0])  # שורה ראשונה


print(df.loc[0, 'Name'])  # שם של השורה הראשונה


