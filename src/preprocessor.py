import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read the dataset from the Google Sheets sheet
data = pd.read_csv('dataset/dataset.csv')

# Select the categorical columns that need encoding
categorical_columns = list(data.columns)

# Create a LabelEncoder object for each categorical column
label_encoders = {}
print(categorical_columns)
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Split the dataset into features (X) and target variable (y)
X = data.drop('How often do you experience digestive discomfort (e.g., bloating, gas, indigestion)?', axis=1)  # Replace 'target_variable' with the name of your target variable column
y = data['How often do you experience digestive discomfort (e.g., bloating, gas, indigestion)?']

# Save the preprocessed dataset to a new CSV file
preprocessed_data = pd.concat([X, y], axis=1)
preprocessed_data.to_csv('dataset/preprocessed_dataset.csv', index=False)
