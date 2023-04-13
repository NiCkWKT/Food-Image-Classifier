print("//////////////////")
print("   <<依加開始>>")
print("//////////////////")
print(" ")
print(" ")

import numpy as np
import random, pickle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Define directories for training, validation, and test sets
Path = "/Users/kimponghung/Desktop/STAT4012 Project/" #To get the saved datas
train = Path + "training"
valid = Path + "validation"

# Define categories for data sets
categories = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]

# Load the data and labels for the training set
train_data = np.load(train + '_data.npy')
train_labels = np.load(train + '_labels.npy')
valid_data = np.load(valid + '_data.npy')
valid_labels = np.load(valid + '_labels.npy')

# Define the hyperparameter grid
param_grid = {
    'C': [0.1],
    'kernel': ['poly'],
    'degree': [4],
    'class_weight': ['balanced'],
    'max_iter' : [10000]
}
print("Hyperparameter grid is found")

# Create a grid search object (Using 5-fold cross-validation)
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
print("5-fold cross-validation is finished")

# Fit the grid search object to the data
grid_search.fit(train_data, train_labels)

# Print the best hyperparameters
print("Best hyperparameters: ", grid_search.best_params_)

# Make predictions with the best model
best_model = grid_search.best_estimator_

# Save the best model to file
print('Saving best model')
with open('model5.sav', 'wb') as file:
    pickle.dump(best_model, file)
print('Saved best model')
print('//////////////////////////////////////////////////')

prediction = best_model.predict(valid_data)
train_accuracy = best_model.score(train_data, train_labels)
test_accuracy = best_model.score(valid_data, valid_labels)

print("Train accuracy: ", train_accuracy)
print("Test accuracy: ", test_accuracy)

# Calculate precision and recall for all classes
report = classification_report(valid_labels, prediction, target_names=categories)

# Print the report for precision, recall, F1-score, support
print(" ")
print(report)

print(" ")
print(" ")
print("//////////////////")
print("   <<依加完結>>")
print("//////////////////")