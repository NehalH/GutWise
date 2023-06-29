import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Read the dataset from the Google Sheets sheet
data = pd.read_csv('dataset/dataset.csv')

# Select the categorical columns that need encoding
categorical_columns = ['What is your age group?','What is your gender?','How would you describe your typical diet?','How often do you consume foods high in fiber (e.g., fruits, vegetables, rice)?','Do you include fermented foods in your diet? (e.g., yogurt, buttermilk, lassi, etc)','On average, how many liters of water do you consume per day?','How often do you consume alcoholic beverages?','Do you engage in regular physical exercise? (e.g., walking, running, workout)','How would you describe your sleep patterns?','How would you rate your stress levels on a daily basis?','How often do you consume allopathic medicines?','How often do you have a bowel movement (passing stool) in a typical week?','How would you describe the consistency of your stool most of the time?','How often do you experience digestive discomfort (e.g., bloating, gas, indigestion)?']

# Create a LabelEncoder object for each categorical column
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Split the dataset into features (X) and target variable (y)
X = data.drop('How often do you experience digestive discomfort (e.g., bloating, gas, indigestion)?', axis=1)  # Replace 'target_variable' with the name of your target variable column
y = data['How often do you experience digestive discomfort (e.g., bloating, gas, indigestion)?']

# Save the preprocessed dataset to a new CSV file
preprocessed_data = pd.concat([X, y], axis=1)
preprocessed_data.to_csv('dataset/preprocessed_dataset.csv', index=False)
