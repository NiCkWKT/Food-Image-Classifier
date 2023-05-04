import numpy as np
import pandas as pd
import pickle

# Define directories for training, validation, and test sets
Path = "/Users/kimponghung/Desktop/STAT4012 Project/Data/" #To get the saved datas
test = Path + "evaluation"

# Define categories for data sets
categories = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]

# Read the CSV file using pandas
test_data = pd.read_csv(Path + "evaluation_data.csv")
test_labels = pd.read_csv(Path + "evaluation_labels.csv")

# Extract the data
test_data = np.array(test_data)
test_labels = np.ravel(test_labels)

model = "SVM_model4"

# Load the saved model
with open( model + '.sav', 'rb') as file:
    best_model = pickle.load(file)

# Make predictions with the best model on the validation set
prediction = best_model.predict(test_data)
test_accuracy = best_model.score(test_data, test_labels)

# Convert the predictions to a DataFrame
pred_df = pd.DataFrame({'Prediction': prediction})

# Save the results to a CSV file
pred_df.to_csv('prediction_' + model + '.csv', index=False)

print("/////////")
print("finished")
print("/////////")