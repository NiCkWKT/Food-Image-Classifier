import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Define directories for training, validation, and test sets
Path = "/Users/kimponghung/Desktop/STAT4012 Project/Prediction/" #To get the saved datas

# Define categories for data sets
categories = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    #getting the standard confusion matrix in text form
    cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred))
    #using the matrix generated as means to plot a confusion matrix graphically
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

model = "alexnet"

tl = "true_labels"
model_path = "prediction_" + model

# Read the CSV file using pandas
pred_df = pd.read_csv(Path + model_path + ".csv")
labels_df = pd.read_csv(Path + tl + ".csv")

# Extract the category labels from the second column
pred = pred_df.iloc[:, 1].values
test_labels = labels_df.iloc[:, 1].values

# Convert the labels to a numpy array
pred = np.array(pred)
test_labels = np.array(test_labels)

# Print the classification report
report = classification_report(test_labels, pred, target_names=categories)
print(report)
with open(model + '_cr.txt', 'w') as f:
    f.write(report)

# Plot the confusion matrix
plot_confusion_matrix(test_labels, pred, classes=categories, normalize=False, title='Confusion matrix for ' + model)
plt.show()
plt.close()

# Plot the normalized confusion matrix
plot_confusion_matrix(test_labels, pred, classes=categories, normalize=True, title='Normalized confusion matrix for ' + model)
plt.show()
plt.close()