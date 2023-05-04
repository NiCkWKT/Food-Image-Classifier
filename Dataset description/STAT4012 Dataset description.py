import os
import csv

Path = "/Users/kimponghung/Desktop/STAT4012 Project/archive/"
train_dir = os.path.join(Path, "training")
valid_dir = os.path.join(Path, "validation")
test_dir = os.path.join(Path, "evaluation")

categories = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]
directories = [train_dir, valid_dir, test_dir]

for directory in directories:
    relative_path = os.path.relpath(directory, Path)
    category_counts = {category: 0 for category in categories}
    for category in categories:
        category_dir = os.path.join(directory, category)
        count = len(os.listdir(category_dir))
        category_counts[category] = count
    total_count = sum(category_counts.values())
    
    # save results to CSV file
    with open(f"{relative_path}_set_category_counts.csv", mode="w") as csv_file:
        fieldnames = ["Category", "Count", "Percentage"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for category, count in category_counts.items():
            percentage = count / total_count * 100
            writer.writerow({"Category": category, "Count": count, "Percentage": f"{percentage:.2f}%"})
    
    with open(f"{relative_path}_set_total_count.csv", mode="w") as csv_file:
        fieldnames = ["Total Count"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"Total Count": total_count})