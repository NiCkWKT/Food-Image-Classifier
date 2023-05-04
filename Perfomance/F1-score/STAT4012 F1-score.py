import matplotlib.pyplot as plt

# Define the class names and f1-score values
class_names = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat',
               'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit']
f1_score = [0.58, 0.55, 0.68, 0.72, 0.72, 0.81, 0.79, 0.69, 0.80, 0.82, 0.88]

report = "XGBoost_best_model"

# Set the figure size and create a horizontal bar chart of f1-score values
plt.figure(figsize=(8, 6))

# Set different colors for bars with f1-score values greater than or equal to 0.8, between 0.2 and 0.8, and less than or equal to 0.2
colors = ['#f03b20' if score <= 0.2 else '#fee391' if score <= 0.8 else '#74c476' for score in f1_score]
plt.barh(class_names, f1_score, height=0.7, color=colors)

# Add labels and title
plt.xlabel('F1-score', fontsize=14)
plt.ylabel('Class', fontsize=14)
plt.title('F1-scores for Food Image Classification for ' + report , fontsize=16)

# Add value labels to the bars
for i, v in enumerate(f1_score):
    plt.text(v + 0.01, i, str(round(v, 2)), fontsize=12)

# Customize the tick marks and gridlines
plt.tick_params(axis='both', which='major', labelsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Display the plot
plt.show()