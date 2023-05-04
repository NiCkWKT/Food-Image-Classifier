import os
import matplotlib.pyplot as plt

Path = "/Users/kimponghung/Desktop/STAT4012 Project/archive/"
train_dir = os.path.join(Path, "training")
valid_dir = os.path.join(Path, "validation")
test_dir = os.path.join(Path, "evaluation")

categories = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]
directories = [train_dir, valid_dir, test_dir]
set_names = ["Training", "Validation", "Evaluation"]

# Calculate category counts and total counts for each directory
category_counts = []
total_counts = []
for directory in directories:
    category_count = {category: 0 for category in categories}
    for category in categories:
        category_dir = os.path.join(directory, category)
        count = len(os.listdir(category_dir))
        category_count[category] = count
    total_count = sum(category_count.values())
    category_counts.append(category_count)
    total_counts.append(total_count)

# Define colors and explode values
colors = plt.cm.Set3.colors
explode = [0.1 if count/total_counts[i] < 0.05 else 0.05 for i, category_count in enumerate(category_counts) for category, count in category_count.items()]

# Define font properties for labels
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }

# Plot pie charts for category counts for each directory
for i, directory in enumerate(directories):
    plt.figure(i)
    plt.title(f"{set_names[i]} Set Category Distribution", fontsize=16)
    labels = [f"{category}\n({category_counts[i][category]})" for category in categories]
    sizes = [category_counts[i][category] for category in categories]

    # Define the explode values for each slice
    explode_i = explode[i*len(categories):(i+1)*len(categories)]

    # Plot the pie chart with explode values, colors, and shadow
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode_i, textprops=font, shadow=True, labeldistance=1.15)

    # Add legend
    plt.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5), labels=[f"{category} ({count})" for category, count in category_counts[i].items()])

# Plot pie chart for total counts for all categories in each directory
plt.figure(3)
plt.title("Total Count Distribution", fontsize=16)
labels = set_names
sizes = total_counts

# Define the explode values for each slice
explode_total = [0.1 if count/sum(total_counts) < 0.05 else 0.05 for count in total_counts]

# Plot the pie chart with explode values, colors, and shadow
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode_total, textprops=font, shadow=True)

# Add legend
plt.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5), labels=[f"{set_name} ({count})" for set_name, count in zip(set_names, total_counts)])

# Plot pie chart for total counts for training and validation sets
plt.figure(4)
plt.title("Training vs Validation Total Count Distribution", fontsize=16)
labels = ["Training", "Validation"]
sizes = [total_counts[0], total_counts[1]]

# Define the explode values for each slice
explode_tv = [0.1 if count/sum(sizes) < 0.05 else 0.05 for count in sizes]

# Plot the pie chart with explode values, colors, and shadow
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode_tv, textprops=font, shadow=True)

# Add legend
plt.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5), labels=[f"{label} ({size})" for label, size in zip(labels, sizes)])

plt.show()