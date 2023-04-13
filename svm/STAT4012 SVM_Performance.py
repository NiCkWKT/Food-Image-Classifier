import numpy as np
import pickle
from sklearn.metrics import classification_report

# Define directories for training, validation, and test sets
Path = "/Users/kimponghung/Desktop/STAT4012 Project/" #To get the saved datas
test = Path + "evaluation"

# Define categories for data sets
categories = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]

# Load the data and labels for the training set
test_data = np.load(test + '_data.npy')
test_labels = np.load(test + '_labels.npy')

# Load the saved model
with open('model4.sav', 'rb') as file:
    best_model = pickle.load(file)

# Make predictions with the best model on the validation set
prediction = best_model.predict(test_data)
test_accuracy = best_model.score(test_data, test_labels)

print("Evaluation accuracy: ", test_accuracy)

# Calculate precision and recall for all classes
report = classification_report(test_labels, prediction, target_names=categories)

# Print the report for precision, recall, F1-score, support
print(" ")
print(report)