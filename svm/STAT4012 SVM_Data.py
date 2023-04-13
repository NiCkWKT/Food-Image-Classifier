import numpy as np
import os, cv2, random

# Define directories for training, validation, and test sets.
Path = "/Users/kimponghung/Desktop/STAT4012 Project/archive/" #To get the original files
train = Path + "training"
valid = Path + "validation"
test = Path + "evaluation"

categories = ["Bread", "Dairy product", "Dessert", "Egg", "Fried food", "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"]
directions = [train, valid, test]

data = []
for dir in directions:
    print("=====================================")
    relative_path = dir[len(Path):]  # extract the part of the directory path after `Path`
    print(f'loading... set : {relative_path}_set')
    print("/////////////////////////////////////")
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
    # Save the data and labels as .npy files
    print(f'saving... set : {relative_path}_set')
    np.save(relative_path + '_data.npy', features)
    np.save(relative_path + '_labels.npy', labels)
    print(f'saved set : {relative_path}_set')