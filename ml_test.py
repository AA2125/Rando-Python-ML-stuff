#Screw C python is king idk if it takes all day to run

# this shit is sort of intersting thnx stack overflow
#python >> C any day every day idk if it takes "Forever" to run.

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

#Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

#Divide the data set into 2 things (training and test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # learn how this stuff works fallow the book.

#Decision Tree classifier is initailzed 
clf = DecisionTreeClassifier()

#Train the classifier
clf.fit(X_train, y_train)

#Make predictions on the test set
y_pred = clf.predict(X_test)

#Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

#print the results
print("Classification Report:")
print(classification_report(y_test, y_pred))

#Random example that I plugged in
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = clf.predict(sample)
print(f"Predicted class for sample {sample}: {iris.target_names[prediction][0]}")

#revise prob and stats bruh u need it.
