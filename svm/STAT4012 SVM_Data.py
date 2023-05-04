import numpy as np
import pandas as pd
import os, cv2, random

# Define directories for training, validation, and test sets.
Path = "/Users/kimponghung/Desktop/STAT4012 Project/archive/" #To get the original files
train = Path + "training"
valid = Path + "validation"
test = Path + "evaluation"

categories = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]
directions = [train]

for dir in directions:
    print("=====================================")
    relative_path = dir[len(Path):]  # extract the part of the directory path after `Path`
    print(f'loading... set : {relative_path}_set')
    print("/////////////////////////////////////")
    
    data = []  # create a new data list for each directory
    for category in categories:
        print(f'loading {relative_path} category : {category}')
        path = os.path.join(dir, category)
        label = categories.index(category)

        for img in os.listdir(path):
            imgpath = os.path.join(path, img)
            food_img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
            food_img = cv2.resize(food_img, (128,128))
            food_img = food_img.astype('float32') / 255.0

            image = np.array(food_img).flatten()
            data.append([image,label])
        print(f'loaded {relative_path} category : {category} successfully')
    
    print("/////////////////////////////////////")
    print(f'loaded set : {relative_path}_set')
    print("=====================================")

    random.shuffle(data)
    features = []
    labels = []

    for feature, label in data:
        features.append(feature)
        labels.append(label)
    
    # Save the data and labels as CSV files with column names
    print(f'saving... set : {relative_path}_set')
    features_df = pd.DataFrame(features, columns=[f'pixel{i}' for i in range(len(features[0]))])
    labels_df = pd.DataFrame(labels, columns=['label'])
    features_df.to_csv(relative_path + '_data.csv', index=False)
    labels_df.to_csv(relative_path + '_labels.csv', index=False)
    print(f'saved set : {relative_path}_set')