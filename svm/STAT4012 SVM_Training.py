import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import multiprocessing as mp

# Define directories for training, validation, and test sets
Path = "/Users/kimponghung/Desktop/STAT4012 Project/Data/" #To get the saved datas

# Define categories for data sets
categories = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]

# Read the CSV file using pandas
train_data = pd.read_csv(Path + "training_data.csv")
train_labels = pd.read_csv(Path + "training_labels.csv")
valid_data = pd.read_csv(Path + "validation_data.csv")
valid_labels = pd.read_csv(Path + "validation_labels.csv")

# Extract the data
train_data = np.array(train_data)
train_labels = np.ravel(train_labels)
valid_data = np.array(valid_data)
valid_labels = np.ravel(valid_labels)

# Define the hyperparameter grid
param_grid = {
    'C': [0.1],
    'kernel': ['poly','rbf','sigmoid'],
    'max_iter' : [10000]
}

# Create a grid search object (Using 5-fold cross-validation)
grid_search = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=3)

# Fit the grid search object to the data
grid_search.fit(train_data, train_labels)

# Print the best hyperparameters
print("Best hyperparameters: ", grid_search.best_params_)

# Make predictions with the best model
best_model = grid_search.best_estimator_

model = "model4"

# Save the best model to file
print('Saving best model')
with open('SVM_' + model + '.sav', 'wb') as file:
    pickle.dump(best_model, file)
print('Saved best model')
print('//////////////////////////////////////////////////')

prediction = best_model.predict(valid_data)
train_accuracy = best_model.score(train_data, train_labels)
test_accuracy = best_model.score(valid_data, valid_labels)

print("Train accuracy: ", train_accuracy)
print("Test accuracy: ", test_accuracy)