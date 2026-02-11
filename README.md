# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, split into features and target, preprocess numerical and categorical data.
2. Create a pipeline combining preprocessing and Decision Tree Classifier.
3. Use GridSearchCV to optimize hyperparameters and train the best model.
4. Evaluate accuracy, classification metrics, confusion matrix, ROC curve, and visualize the decision tree.

## Program:
```
#  Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Create simple dataset
data = {
    "Maths": [35, 78, 90, 45, 20, 60, 55, 30],
    "Science": [40, 85, 88, 50, 25, 65, 58, 35],
    "English": [45, 80, 92, 48, 30, 70, 60, 38],
    "Result": ["Pass", "Pass", "Pass", "Pass", "Fail", "Pass", "Pass", "Fail"]
}

df = pd.DataFrame(data)

X = df[["Maths", "Science", "English"]]
y = df["Result"]

#  Train Decision Tree
model = DecisionTreeClassifier(criterion="gini", max_depth=3)
model.fit(X, y)

#  Plot Decision Tree
plt.figure(figsize=(14, 8))
plot_tree(
    model,
    feature_names=["Maths", "Science", "English"],
    class_names=["Fail", "Pass"],
    filled=True
)
plt.title("Decision Tree: Student Result (Pass/Fail)")
plt.show()
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Mukesh M
RegisterNumber:  212225240093
*/
```

## Output:
![2ad898cc-6a21-4a80-8e63-c50e5b507548](https://github.com/user-attachments/assets/c34e4acb-a349-4297-bd06-e690922d0b7f)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
