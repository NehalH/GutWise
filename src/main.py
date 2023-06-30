import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

target_variable = 'How often do you experience digestive discomfort (e.g., bloating, gas, indigestion)?'

# Load the Dataset
data = pd.read_csv('dataset/preprocessed_dataset.csv')  # Replace 'your_dataset.csv' with the path to your dataset

# Split the Data into Training and Testing Sets
X = data.drop(target_variable, axis=1)  # Replace 'target_variable' with the name of your target variable column
y = data[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a Classification Model
model = DecisionTreeClassifier()

# Train the Model
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate Performance and Construct Confusion Matrix
# confusion_mat = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:")
# print(confusion_mat)

# Calculate Performance Metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, average='weighted')  # Change the average parameter to 'weighted'
recall = recall_score(y_test, y_pred, average='weighted')  # Change the average parameter to 'weighted'
f_measure = f1_score(y_test, y_pred, average='weighted')  # Change the average parameter to 'weighted'

print("Accuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)
print("F-Measure:", f_measure)
