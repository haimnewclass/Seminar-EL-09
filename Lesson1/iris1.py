# שלב 1: טעינת ספריות
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# שלב 2: טעינת הדאטהסט של Iris
data = load_iris()
X = data.data  # המאפיינים
y = data.target  # התגים

# שלב 3: חלוקת הדאטהסט לאימון ולמבחן
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# שלב 4: נרמול הנתונים
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# שלב 5: המרת הנתונים ל-Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# שלב 6: יצירת TensorDataset
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# שלב 7: יצירת DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


# שלב 8: הגדרת רשת נוירונים
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x


# שלב 9: אתחול הרשת
input_size = X_train.shape[1]  # מספר המאפיינים
hidden_size = 10  # מספר הנוירונים בשכבה החבויה
output_size = len(data.target_names)  # מספר המחלקות
model = NeuralNet(input_size, hidden_size, output_size)

# שלב 10: בחירת פונקציית הפסד ואופטימייזר
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# שלב 11: אימון המודל
epochs = 50
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()  # אפס את הגרדיאנטים
        outputs = model(X_batch)  # תחזיות
        loss = criterion(outputs, y_batch)  # חישוב הפסד
        loss.backward()  # חישוב גרדיאנטים
        optimizer.step()  # עדכון הפרמטרים
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# שלב 12: הערכת המודל
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)  # חיזוי המחלקה עם הסתברות הגבוהה ביותר
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
